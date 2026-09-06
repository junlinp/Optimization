#include "eskf_estimator.h"

namespace vio {
namespace {
using Matrix15 = Eigen::Matrix<double, 15, 15>;
using Matrix15x12 = Eigen::Matrix<double, 15, 12>;
using Matrix12 = Eigen::Matrix<double, 12, 12>;
using Matrix6x15 = Eigen::Matrix<double, 6, 15>;
using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Vector6 = Eigen::Matrix<double, 6, 1>;
}  // namespace

EskfEstimator::EskfEstimator(const EskfState& initial_state,
                             const Eigen::Matrix<double, 15, 15>& initial_covariance,
                             const ImuNoiseParams& noise_params,
                             const Eigen::Vector3d& gravity_world)
    : state_(initial_state),
      P_(initial_covariance),
      noise_(noise_params),
      gravity_world_(gravity_world) {}

void EskfEstimator::Predict(const Eigen::Vector3d& gyro_meas,
                            const Eigen::Vector3d& accel_meas, double dt) {
  const Eigen::Vector3d omega = gyro_meas - state_.bias_gyro;
  const Eigen::Vector3d a = accel_meas - state_.bias_accel;
  const Sophus::SO3d R_k = state_.R;
  const Eigen::Matrix3d R_k_mat = R_k.matrix();
  const Eigen::Vector3d a_world = R_k_mat * a + gravity_world_;
  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d a_skew = Sophus::SO3d::hat(a);

  // Error-state transition Jacobian F_x. See eskf_estimator.h and the plan
  // (vio/BUILD-adjacent design doc) for the derivation; the dtheta/dtheta
  // block is the SO(3) adjoint of the (negative of the) rotation increment,
  // everything else is unlisted-zero except the diagonal identity blocks.
  Matrix15 F_x = Matrix15::Identity();
  F_x.block<3, 3>(0, 3) = I3 * dt;
  F_x.block<3, 3>(0, 6) = -0.5 * R_k_mat * a_skew * dt * dt;
  F_x.block<3, 3>(0, 12) = -0.5 * R_k_mat * dt * dt;
  F_x.block<3, 3>(3, 6) = -R_k_mat * a_skew * dt;
  F_x.block<3, 3>(3, 12) = -R_k_mat * dt;
  F_x.block<3, 3>(6, 6) = Sophus::SO3d::exp(-omega * dt).matrix();
  F_x.block<3, 3>(6, 9) = -I3 * dt;

  // Noise Jacobian F_i and process noise Q_i. Noise vector order is
  // [n_g(0:3), n_a(3:6), n_bg(6:9), n_ba(9:12)]. F_i's coefficients carry no
  // dt; Q_i is linear in dt (Var[integral of white noise over dt] = sigma^2
  // * dt) -- see the design doc's dimensional-analysis derivation. Getting
  // this dt-scaling right (not dt^2, not sigma^2/dt) is the single most
  // common ESKF implementation bug, which is exactly what
  // PredictSingleStepCovarianceMatchesClosedForm below regression-tests.
  Matrix15x12 F_i = Matrix15x12::Zero();
  F_i.block<3, 3>(3, 3) = -R_k_mat;
  F_i.block<3, 3>(6, 0) = -I3;
  F_i.block<3, 3>(9, 6) = I3;
  F_i.block<3, 3>(12, 9) = I3;

  Matrix12 Q_i = Matrix12::Zero();
  Q_i.block<3, 3>(0, 0) = (noise_.gyro_noise_density * noise_.gyro_noise_density) * I3;
  Q_i.block<3, 3>(3, 3) = (noise_.accel_noise_density * noise_.accel_noise_density) * I3;
  Q_i.block<3, 3>(6, 6) = (noise_.gyro_random_walk * noise_.gyro_random_walk) * I3;
  Q_i.block<3, 3>(9, 9) = (noise_.accel_random_walk * noise_.accel_random_walk) * I3;
  Q_i *= dt;

  P_ = F_x * P_ * F_x.transpose() + F_i * Q_i * F_i.transpose();

  // Nominal state propagation (strapdown mechanization), using R_k (the
  // pre-update rotation) throughout -- must happen after the Jacobians
  // above, which are built from R_k, not R_{k+1}.
  state_.p = state_.p + state_.v * dt + 0.5 * a_world * dt * dt;
  state_.v = state_.v + a_world * dt;
  state_.R = R_k * Sophus::SO3d::exp(omega * dt);
  // biases held constant in the nominal propagation; only their covariance grows.
}

void EskfEstimator::SetRelativePoseAnchor() {
  anchor_R_ = state_.R;
  anchor_p_ = state_.p;
  have_anchor_ = true;
}

bool EskfEstimator::UpdateRelativePose(const RelativePoseMeasurement& measurement) {
  if (!have_anchor_) return false;

  // T_h = T_k^{-1} * T_{k+1}, the relative pose predicted by the filter's
  // own propagated nominal states at the anchor and now.
  const Sophus::SO3d R_h = anchor_R_.inverse() * state_.R;
  const Eigen::Vector3d t_h = anchor_R_.inverse() * (state_.p - anchor_p_);
  const Sophus::SE3d T_h(R_h, t_h);

  // Measurement Jacobian H (6x15), rows [upsilon(0:3); omega(3:6)], columns
  // [dp(0:3),dv(3:6),dtheta(6:9),db_g(9:12),db_a(12:15)]. Only frame k+1's
  // dp/dtheta columns are nonzero -- see the design doc for the derivation
  // and the two documented simplifications (T_k fixed; SE(3) left-Jacobian
  // correction approximated as identity) this rests on.
  Matrix6x15 H = Matrix6x15::Zero();
  H.block<3, 3>(0, 0) = state_.R.inverse().matrix();
  H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();

  // Standard Lie-group EKF residual convention: predicted^{-1} * measured
  // (boxminus), NOT the reverse. Swapping this order flips the sign of the
  // correction's effect on the linearized residual (dr/ddx goes from -H to
  // +H), which turns UpdateRelativePose into an error-amplifying update
  // instead of a correcting one -- caught by
  // UpdateInjectedErrorCorrectsTowardTruth below, which failed with the
  // reversed order (state moved twice as far from truth, not toward it).
  const Vector6 r = (T_h.inverse() * measurement.T_prevbody_currbody).log();

  Matrix6 R6 = Matrix6::Zero();
  R6.block<3, 3>(0, 0) =
      (measurement.sigma_translation_m * measurement.sigma_translation_m) *
      Eigen::Matrix3d::Identity();
  R6.block<3, 3>(3, 3) = (measurement.sigma_rotation_rad * measurement.sigma_rotation_rad) *
                        Eigen::Matrix3d::Identity();

  const Matrix6 S = H * P_ * H.transpose() + R6;
  const Eigen::Matrix<double, 15, 6> K = P_ * H.transpose() * S.inverse();
  const Eigen::Matrix<double, 15, 1> dx = K * r;

  state_.p += dx.segment<3>(0);
  state_.v += dx.segment<3>(3);
  state_.R = state_.R * Sophus::SO3d::exp(dx.segment<3>(6));
  state_.bias_gyro += dx.segment<3>(9);
  state_.bias_accel += dx.segment<3>(12);

  P_ = (Matrix15::Identity() - K * H) * P_;
  return true;
}

}  // namespace vio

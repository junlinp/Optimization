#include "eqf_star_attitude.h"

namespace eqf_star_attitude {

Eigen::Matrix3d EqfStarAttitude::EquivariantOutputBlock(
    const Eigen::Vector3d& y_meas, const Eigen::Vector3d& y_pred,
    const Eigen::Matrix3d& /*R_hat*/) {
  // Full SO(3) body-vector outputs: midpoint Jacobian without S^2 projection.
  return 0.5 * (Sophus::SO3d::hat(y_meas) + Sophus::SO3d::hat(y_pred));
}

EqfStarAttitude::EqfStarAttitude(
    const EqfStarAttitudeState& initial_state,
    const Eigen::Matrix<double, 3, 3>& initial_covariance,
    const EqfStarImuNoiseParams& noise_params,
    const Eigen::Vector3d& gravity_world_reference,
    const Eigen::Vector3d& magnetometer_world_reference)
    : state_(initial_state),
      sigma_(initial_covariance),
      P_(Eigen::Matrix3d::Identity() * 1e-6),
      noise_(noise_params),
      dm_(magnetometer_world_reference.normalized()),
      dg_(gravity_world_reference.normalized()) {}

void EqfStarAttitude::Predict(const Eigen::Vector3d& gyro_meas, double dt) {
  state_.R = state_.R * Sophus::SO3d::exp(gyro_meas * dt);
  sigma_ += P_ * dt;
}

void EqfStarAttitude::UpdateVector(const VectorMeasurement& gravity,
                                   const VectorMeasurement& magnetometer,
                                   double /*dt*/) {
  Eigen::Matrix<double, 6, 1> y_hat;
  y_hat << state_.R.inverse() * dg_, state_.R.inverse() * dm_;

  Eigen::Matrix<double, 6, 1> y;
  y << gravity.body_measured, magnetometer.body_measured;

  const Eigen::Matrix<double, 6, 1> r = y - y_hat;

  const Eigen::Matrix3d R_hat = state_.R.matrix();
  Eigen::Matrix<double, 6, 3> C;
  C.block<3, 3>(0, 0) =
      EquivariantOutputBlock(y.segment<3>(0), y_hat.segment<3>(0), R_hat);
  C.block<3, 3>(3, 0) =
      EquivariantOutputBlock(y.segment<3>(3), y_hat.segment<3>(3), R_hat);

  Q_.setZero();
  Q_.block<3, 3>(0, 0) =
      (gravity.sigma * gravity.sigma) * Eigen::Matrix3d::Identity();
  Q_.block<3, 3>(3, 3) =
      (magnetometer.sigma * magnetometer.sigma) * Eigen::Matrix3d::Identity();

  const Eigen::Matrix<double, 6, 6> S = C * sigma_ * C.transpose() + Q_;
  const Eigen::Matrix<double, 3, 6> K = sigma_ * C.transpose() * S.inverse();
  // EqF (26): dL => R*exp(Omega dt); dR => exp(Delta)*R, Delta = R_hat*K*r.
  const Eigen::Vector3d delta = state_.R * (K * r);
  state_.R = Sophus::SO3d::exp(delta) * state_.R;

  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d IKC = I3 - K * C;
  sigma_ = IKC * sigma_ * IKC.transpose() + K * Q_ * K.transpose();
}

}  // namespace eqf_star_attitude

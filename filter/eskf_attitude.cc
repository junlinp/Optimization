#include "eskf_attitude.h"

namespace attitude_filter {
namespace {
using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Matrix3x6 = Eigen::Matrix<double, 3, 6>;
using Vector6 = Eigen::Matrix<double, 6, 1>;
}  // namespace

AttitudeEskf::AttitudeEskf(const AttitudeEskfState& initial_state,
                          const Matrix6& initial_covariance,
                          const AttitudeImuNoiseParams& noise_params)
    : state_(initial_state), P_(initial_covariance), noise_(noise_params) {}

void AttitudeEskf::Predict(const Eigen::Vector3d& gyro_meas, double dt) {
  const Eigen::Vector3d omega = gyro_meas - state_.bias_gyro;
  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();

  // Same [dtheta, db_g] block structure as the dtheta/db_g columns of
  // vio::EskfEstimator::Predict's 15x15 F_x/F_i -- see that file for the
  // derivation.
  Matrix6 F_x = Matrix6::Identity();
  F_x.block<3, 3>(0, 0) = Sophus::SO3d::exp(-omega * dt).matrix();
  F_x.block<3, 3>(0, 3) = -I3 * dt;

  Matrix6 F_i = Matrix6::Zero();
  F_i.block<3, 3>(0, 0) = -I3;
  F_i.block<3, 3>(3, 3) = I3;

  Matrix6 Q_i = Matrix6::Zero();
  Q_i.block<3, 3>(0, 0) = (noise_.gyro_noise_density * noise_.gyro_noise_density) * I3;
  Q_i.block<3, 3>(3, 3) = (noise_.gyro_random_walk * noise_.gyro_random_walk) * I3;
  Q_i *= dt;

  P_ = F_x * P_ * F_x.transpose() + F_i * Q_i * F_i.transpose();

  state_.R = state_.R * Sophus::SO3d::exp(omega * dt);
  // bias held constant in the nominal propagation; only its covariance grows.
}

void AttitudeEskf::UpdateVector(const VectorMeasurement& measurement) {
  // Predicted body-frame vector from the current nominal rotation.
  const Eigen::Vector3d h0 = state_.R.inverse() * measurement.world_reference;

  // For R_true=R*Exp(dtheta): h(dtheta) = Exp(-dtheta)*R^{-1}*v_world
  // ~= h0 - skew(dtheta)*h0 = h0 + skew(h0)*dtheta, so dh/ddtheta = skew(h0).
  // The db_g columns are zero -- a vector observation has no direct gyro-bias
  // sensitivity, only an indirect one accumulated through Predict.
  Matrix3x6 H = Matrix3x6::Zero();
  H.block<3, 3>(0, 0) = Sophus::SO3d::hat(h0);

  const Eigen::Vector3d r = measurement.body_measured - h0;
  const Eigen::Matrix3d R_meas = (measurement.sigma * measurement.sigma) *
                                Eigen::Matrix3d::Identity();

  const Eigen::Matrix3d S = H * P_ * H.transpose() + R_meas;
  const Eigen::Matrix<double, 6, 3> K = P_ * H.transpose() * S.inverse();
  const Vector6 dx = K * r;

  state_.R = state_.R * Sophus::SO3d::exp(dx.segment<3>(0));
  state_.bias_gyro += dx.segment<3>(3);

  P_ = (Matrix6::Identity() - K * H) * P_;
}

}  // namespace attitude_filter

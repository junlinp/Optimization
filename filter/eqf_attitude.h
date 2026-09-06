#ifndef EQF_ATTITUDE_H
#define EQF_ATTITUDE_H
// Standard EqF for SO(3) attitude (van Goor et al., IEEE TAC 2022 / arXiv:2107.05193).
// Observer (26): X_dot = dL_X Lambda + dR_X Delta.
// Predict (dL): R <- R*exp(Omega dt).  Update (dR): R <- exp(Delta)*R.
// Error coords eps = log(R_hat^{-1} R); discrete Delta = R_hat * K * r.
#include <Eigen/Dense>
#include "sophus/so3.hpp"

namespace eqf_attitude {

struct EqfAttitudeState {
  Sophus::SO3d R;  // R_world_body, defaults to identity
  Eigen::Vector3d bias_gyro = Eigen::Vector3d::Zero();
};

struct EqfImuNoiseParams {
  double gyro_noise_density = 0;  // rad/s / sqrt(Hz)
  double gyro_random_walk = 0;    // rad/s^2 / sqrt(Hz)
};

struct VectorMeasurement {
  Eigen::Vector3d body_measured = Eigen::Vector3d::Zero();
  double sigma = 0.01;
};

class EqfAttitude {
 public:
  EqfAttitude(const EqfAttitudeState& initial_state,
              const Eigen::Matrix<double, 3, 3>& initial_covariance,
              const EqfImuNoiseParams& noise_params,
              const Eigen::Vector3d& gravity_world_reference,
              const Eigen::Vector3d& magnetometer_world_reference);
  ~EqfAttitude() = default;

  void Predict(const Eigen::Vector3d& gyro_meas, double dt);
  void UpdateVector(const VectorMeasurement& gravity,
                    const VectorMeasurement& magnetometer,
                    double dt);

  const EqfAttitudeState& state() const { return state_; }

 private:
  EqfAttitudeState state_;
  Eigen::Matrix<double, 3, 3> sigma_;
  Eigen::Matrix<double, 3, 3> P_;
  Eigen::Matrix<double, 6, 6> Q_;
  EqfImuNoiseParams noise_;
  Eigen::Vector3d dm_;  // known magnetometer direction in world frame
  Eigen::Vector3d dg_;  // known gravity direction in world frame
};

}  // namespace eqf_attitude
#endif  // EQF_ATTITUDE_H

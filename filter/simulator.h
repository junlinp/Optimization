#ifndef ATTITUDE_FILTER_SIMULATOR_H_
#define ATTITUDE_FILTER_SIMULATOR_H_
#include <cstdint>
#include <vector>

#include <Eigen/Dense>

#include "sophus/so3.hpp"

namespace attitude_filter {

// Ported from the user-supplied attitude_simulator.cpp: a rigid body
// tumbling under a smooth analytic angular-velocity profile, observed by a
// noisy gyro plus noisy body-frame gravity and magnetometer vector sensors.
struct SimulatorConfig {
  double duration_s = 30.0;
  double frequency_hz = 100.0;
  double gyro_noise_std = 0.005;         // rad/s, additive per sample (not a rate density)
  double gyro_bias_rw = 0.0002;          // rad/s/sqrt(s), continuous random-walk density
  double magnetometer_noise_std = 0.01;  // added to the normalized body-frame vector
  double gravity_noise_std = 0.01;       // added to the normalized body-frame vector
  bool use_gyro_bias = false;
  std::uint32_t seed = 42;
};

// One simulated instant. R_true follows the vio::EskfState convention:
// R_world_body, i.e. Rotate(R_true, v_body) = v_world.
struct SimulatorSample {
  double t = 0.0;
  Sophus::SO3d R_true;
  Eigen::Vector3d omega_true = Eigen::Vector3d::Zero();
  Eigen::Vector3d omega_meas = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_bias_true = Eigen::Vector3d::Zero();
  Eigen::Vector3d mag_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d mag_body_true = Eigen::Vector3d::Zero();
  Eigen::Vector3d mag_body_meas = Eigen::Vector3d::Zero();
  Eigen::Vector3d gravity_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d gravity_body_true = Eigen::Vector3d::Zero();
  Eigen::Vector3d gravity_body_meas = Eigen::Vector3d::Zero();
};

std::vector<SimulatorSample> Simulate(const SimulatorConfig& config);

}  // namespace attitude_filter
#endif  // ATTITUDE_FILTER_SIMULATOR_H_

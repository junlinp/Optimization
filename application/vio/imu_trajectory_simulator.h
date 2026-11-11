#ifndef VIO_IMU_TRAJECTORY_SIMULATOR_H_
#define VIO_IMU_TRAJECTORY_SIMULATOR_H_
#include <cstdint>
#include <vector>

#include <Eigen/Dense>

#include "eskf_estimator.h"
#include "sophus/so3.hpp"

namespace vio {

// Circular motion in the xy-plane with a cosine bob on z, body frame yawing
// to stay tangent to the circle (roll/pitch held at zero, so gyro_true is
// constant). This is the trajectory family used by IMU-preintegration
// consistency benchmarks such as Tsao & Jan 2023 and Delama et al. 2024
// (arXiv:2411.05548): neither paper publishes the exact radius/rate values,
// so treat this as a standard reconstruction of the pattern, not a verbatim
// replica.
struct ImuTrajectoryConfig {
  double duration_s = 30.0;
  double frequency_hz = 200.0;  // IMU sample rate.

  double radius_m = 5.0;            // R
  double circle_rate_rad_s = 0.18;  // omega_c; orbital speed = radius_m * circle_rate_rad_s.
  double z_amplitude_m = 1.0;       // A
  double z_rate_rad_s = 0.6;        // omega_z

  Eigen::Vector3d gravity_world = Eigen::Vector3d(0, 0, -9.81);

  ImuNoiseParams noise;
  bool add_measurement_noise = true;
  bool add_bias_random_walk = false;
  Eigen::Vector3d initial_bias_gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d initial_bias_accel = Eigen::Vector3d::Zero();

  std::uint32_t seed = 42;
};

// One simulated instant: ground truth navigation state plus the raw (bias-
// and noise-corrupted) IMU readings an estimator would consume. R_true
// follows EskfState's convention: R_world_body.
struct ImuTrajectorySample {
  double t = 0.0;

  Eigen::Vector3d p_true = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_true = Eigen::Vector3d::Zero();
  Sophus::SO3d R_true;

  Eigen::Vector3d gyro_true = Eigen::Vector3d::Zero();   // body-frame angular velocity.
  Eigen::Vector3d accel_true = Eigen::Vector3d::Zero();  // body-frame specific force.

  Eigen::Vector3d gyro_bias_true = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel_bias_true = Eigen::Vector3d::Zero();

  Eigen::Vector3d gyro_meas = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel_meas = Eigen::Vector3d::Zero();
};

std::vector<ImuTrajectorySample> SimulateImuTrajectory(const ImuTrajectoryConfig& config);

// Scales every continuous-time noise density in `noise` by `lambda`, matching
// the noise-level multiplier used in the Delama et al. benchmark (e.g.
// lambda in {0.1, 1, 10} for low/medium/high noise).
ImuNoiseParams ScaleNoise(const ImuNoiseParams& noise, double lambda);

}  // namespace vio
#endif  // VIO_IMU_TRAJECTORY_SIMULATOR_H_

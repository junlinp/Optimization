#include "imu_trajectory_simulator.h"

#include <cmath>
#include <random>
#include <stdexcept>

namespace vio {
namespace {

Eigen::Vector3d Gaussian(std::mt19937& rng, double stddev) {
  if (stddev <= 0.0) return Eigen::Vector3d::Zero();
  std::normal_distribution<double> n(0.0, stddev);
  return {n(rng), n(rng), n(rng)};
}

}  // namespace

std::vector<ImuTrajectorySample> SimulateImuTrajectory(const ImuTrajectoryConfig& config) {
  if (config.duration_s <= 0.0 || config.frequency_hz <= 0.0) {
    throw std::invalid_argument("duration and frequency must be positive");
  }

  const double dt = 1.0 / config.frequency_hz;
  const std::size_t count =
      static_cast<std::size_t>(std::floor(config.duration_s * config.frequency_hz)) + 1;

  // Discrete measurement-noise sigma = continuous density / sqrt(dt); discrete
  // bias-random-walk step sigma = continuous density * sqrt(dt). Same
  // convention as EskfEstimator's ImuNoiseParams (see eskf_estimator.h) and
  // filter::Simulate.
  const double gyro_meas_sigma = config.noise.gyro_noise_density / std::sqrt(dt);
  const double accel_meas_sigma = config.noise.accel_noise_density / std::sqrt(dt);
  const double gyro_rw_sigma = config.noise.gyro_random_walk * std::sqrt(dt);
  const double accel_rw_sigma = config.noise.accel_random_walk * std::sqrt(dt);

  const double wc = config.circle_rate_rad_s;
  const double wz = config.z_rate_rad_s;

  std::mt19937 rng(config.seed);
  Eigen::Vector3d gyro_bias = config.initial_bias_gyro;
  Eigen::Vector3d accel_bias = config.initial_bias_accel;

  std::vector<ImuTrajectorySample> samples;
  samples.reserve(count);

  for (std::size_t k = 0; k < count; ++k) {
    const double t = static_cast<double>(k) * dt;
    const double theta = wc * t;
    const double cos_t = std::cos(theta);
    const double sin_t = std::sin(theta);
    const double z_phase = wz * t;
    const double cos_z = std::cos(z_phase);
    const double sin_z = std::sin(z_phase);

    ImuTrajectorySample s;
    s.t = t;
    s.p_true = Eigen::Vector3d(config.radius_m * cos_t, config.radius_m * sin_t,
                                config.z_amplitude_m * cos_z);
    s.v_true = Eigen::Vector3d(-config.radius_m * wc * sin_t, config.radius_m * wc * cos_t,
                                -config.z_amplitude_m * wz * sin_z);
    const Eigen::Vector3d a_world(-config.radius_m * wc * wc * cos_t,
                                   -config.radius_m * wc * wc * sin_t,
                                   -config.z_amplitude_m * wz * wz * cos_z);

    // Body frame yaws to stay tangent to the circle; roll/pitch held at
    // zero, so the true angular velocity is the constant orbital rate about
    // the world/body z-axis.
    s.R_true = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, theta));
    s.gyro_true = Eigen::Vector3d(0, 0, wc);

    // Specific force in the body frame, consistent with EskfEstimator's
    // Predict: a_world = R*(accel_meas-bias) + gravity_world.
    s.accel_true = s.R_true.inverse() * (a_world - config.gravity_world);

    s.gyro_bias_true = gyro_bias;
    s.accel_bias_true = accel_bias;

    s.gyro_meas = s.gyro_true + gyro_bias;
    s.accel_meas = s.accel_true + accel_bias;
    if (config.add_measurement_noise) {
      s.gyro_meas += Gaussian(rng, gyro_meas_sigma);
      s.accel_meas += Gaussian(rng, accel_meas_sigma);
    }

    samples.push_back(s);

    if (config.add_bias_random_walk && k + 1 < count) {
      gyro_bias += Gaussian(rng, gyro_rw_sigma);
      accel_bias += Gaussian(rng, accel_rw_sigma);
    }
  }

  return samples;
}

ImuNoiseParams ScaleNoise(const ImuNoiseParams& noise, double lambda) {
  ImuNoiseParams scaled = noise;
  scaled.gyro_noise_density *= lambda;
  scaled.accel_noise_density *= lambda;
  scaled.gyro_random_walk *= lambda;
  scaled.accel_random_walk *= lambda;
  return scaled;
}

}  // namespace vio

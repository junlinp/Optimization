#include "simulator.h"

#include <cmath>
#include <random>
#include <stdexcept>

namespace attitude_filter {
namespace {

constexpr double kPi = 3.14159265358979323846;

Eigen::Vector3d TrueAngularVelocity(double t) {
  return {0.35 * std::sin(0.7 * t), 0.25 * std::cos(0.5 * t), 0.20 + 0.15 * std::sin(0.3 * t)};
}

Eigen::Vector3d Gaussian(std::mt19937& rng, double stddev) {
  std::normal_distribution<double> n(0.0, stddev);
  return {n(rng), n(rng), n(rng)};
}

}  // namespace

std::vector<SimulatorSample> Simulate(const SimulatorConfig& config) {
  if (config.duration_s <= 0.0 || config.frequency_hz <= 0.0) {
    throw std::invalid_argument("duration and frequency must be positive");
  }

  const double dt = 1.0 / config.frequency_hz;
  const std::size_t count =
      static_cast<std::size_t>(std::floor(config.duration_s * config.frequency_hz)) + 1;

  // Same composed initial attitude as the original tool: Rz(30)*Ry(-20)*Rx(15).
  Sophus::SO3d R_true = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, 30.0 * kPi / 180.0)) *
                       Sophus::SO3d::exp(Eigen::Vector3d(0, -20.0 * kPi / 180.0, 0)) *
                       Sophus::SO3d::exp(Eigen::Vector3d(15.0 * kPi / 180.0, 0, 0));

  const Eigen::Vector3d mag_world = Eigen::Vector3d(0.45, 0.05, 0.89).normalized();
  const Eigen::Vector3d gravity_world = Eigen::Vector3d(0.0, 0.0, -1.0).normalized();

  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
  std::mt19937 rng(config.seed);

  std::vector<SimulatorSample> samples;
  samples.reserve(count);

  for (std::size_t k = 0; k < count; ++k) {
    SimulatorSample s;
    s.t = static_cast<double>(k) * dt;
    s.R_true = R_true;
    s.omega_true = TrueAngularVelocity(s.t);
    s.gyro_bias_true = gyro_bias;
    s.omega_meas = s.omega_true + gyro_bias + Gaussian(rng, config.gyro_noise_std);

    // R_true maps body -> world, so R_true^{-1} maps world -> body.
    s.mag_world = mag_world;
    s.mag_body_true = R_true.inverse() * mag_world;
    s.mag_body_meas =
        (s.mag_body_true + Gaussian(rng, config.magnetometer_noise_std)).normalized();

    s.gravity_world = gravity_world;
    s.gravity_body_true = R_true.inverse() * gravity_world;
    s.gravity_body_meas =
        (s.gravity_body_true + Gaussian(rng, config.gravity_noise_std)).normalized();

    samples.push_back(s);

    if (k + 1 < count) {
      // R_{k+1} = R_k * Exp(omega_body * dt).
      R_true = R_true * Sophus::SO3d::exp(s.omega_true * dt);
      if (config.use_gyro_bias) {
        gyro_bias += Gaussian(rng, config.gyro_bias_rw * std::sqrt(dt));
      }
    }
  }

  return samples;
}

}  // namespace attitude_filter

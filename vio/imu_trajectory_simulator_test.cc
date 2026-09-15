#include "imu_trajectory_simulator.h"

#include "gtest/gtest.h"

namespace vio {
namespace {

ImuNoiseParams SampleNoise() {
  ImuNoiseParams n;
  n.gyro_noise_density = 1.6968e-04;
  n.gyro_random_walk = 1.9393e-05;
  n.accel_noise_density = 2.0000e-3;
  n.accel_random_walk = 3.0000e-3;
  return n;
}

TEST(ImuTrajectorySimulator, ProducesExpectedSampleCount) {
  ImuTrajectoryConfig config;
  config.duration_s = 1.0;
  config.frequency_hz = 100.0;

  const std::vector<ImuTrajectorySample> samples = SimulateImuTrajectory(config);
  EXPECT_EQ(samples.size(), 101u);
  EXPECT_NEAR(samples.front().t, 0.0, 1e-12);
  EXPECT_NEAR(samples.back().t, 1.0, 1e-9);
}

TEST(ImuTrajectorySimulator, PositionStaysOnTheCommandedCircleAndZWave) {
  ImuTrajectoryConfig config;
  config.duration_s = 10.0;
  config.frequency_hz = 100.0;
  config.radius_m = 3.0;
  config.circle_rate_rad_s = 0.4;
  config.z_amplitude_m = 0.5;
  config.z_rate_rad_s = 1.1;

  for (const ImuTrajectorySample& s : SimulateImuTrajectory(config)) {
    EXPECT_NEAR(s.p_true.head<2>().norm(), config.radius_m, 1e-9);
    EXPECT_LE(std::abs(s.p_true.z()), config.z_amplitude_m + 1e-9);
  }
}

// With no noise, no bias, and gravity handled internally, the true specific
// force fed back through R*(accel-bias)+g must reproduce the commanded world
// acceleration -- i.e. the truth channel is self-consistent with the same
// strapdown equation EskfEstimator::Predict uses.
TEST(ImuTrajectorySimulator, TrueSpecificForceIsConsistentWithStrapdownEquation) {
  ImuTrajectoryConfig config;
  config.duration_s = 5.0;
  config.frequency_hz = 50.0;
  config.radius_m = 5.0;
  config.circle_rate_rad_s = 0.3;
  config.z_amplitude_m = 1.0;
  config.z_rate_rad_s = 0.7;
  config.add_measurement_noise = false;

  const std::vector<ImuTrajectorySample> samples = SimulateImuTrajectory(config);
  for (size_t i = 1; i + 1 < samples.size(); ++i) {
    const double dt = 1.0 / config.frequency_hz;
    const Eigen::Vector3d v_dot_numeric = (samples[i + 1].v_true - samples[i - 1].v_true) / (2 * dt);
    const Eigen::Vector3d a_world_reconstructed =
        samples[i].R_true * samples[i].accel_true + config.gravity_world;
    EXPECT_LT((v_dot_numeric - a_world_reconstructed).norm(), 1e-4);
  }
}

TEST(ImuTrajectorySimulator, GyroMatchesConstantOrbitalRate) {
  ImuTrajectoryConfig config;
  config.duration_s = 2.0;
  config.frequency_hz = 100.0;
  config.circle_rate_rad_s = 0.5;
  config.add_measurement_noise = false;

  for (const ImuTrajectorySample& s : SimulateImuTrajectory(config)) {
    EXPECT_LT((s.gyro_meas - Eigen::Vector3d(0, 0, 0.5)).norm(), 1e-9);
  }
}

TEST(ImuTrajectorySimulator, BiasStaysZeroWhenRandomWalkDisabled) {
  ImuTrajectoryConfig config;
  config.duration_s = 5.0;
  config.frequency_hz = 100.0;
  config.noise = SampleNoise();
  config.add_bias_random_walk = false;

  for (const ImuTrajectorySample& s : SimulateImuTrajectory(config)) {
    EXPECT_EQ(s.gyro_bias_true, Eigen::Vector3d::Zero());
    EXPECT_EQ(s.accel_bias_true, Eigen::Vector3d::Zero());
  }
}

TEST(ImuTrajectorySimulator, BiasWalksWhenEnabled) {
  ImuTrajectoryConfig config;
  config.duration_s = 30.0;
  config.frequency_hz = 100.0;
  config.noise.gyro_random_walk = 0.01;  // large, so it moves measurably within the run.
  config.add_bias_random_walk = true;

  const std::vector<ImuTrajectorySample> samples = SimulateImuTrajectory(config);
  EXPECT_GT(samples.back().gyro_bias_true.norm(), 1e-6);
}

TEST(ImuTrajectorySimulator, MeasurementNoiseIsAddedWhenEnabled) {
  ImuTrajectoryConfig config;
  config.duration_s = 5.0;
  config.frequency_hz = 100.0;
  config.noise = SampleNoise();
  config.add_measurement_noise = true;

  bool saw_nonzero_noise = false;
  for (const ImuTrajectorySample& s : SimulateImuTrajectory(config)) {
    if ((s.gyro_meas - s.gyro_true).norm() > 1e-9) saw_nonzero_noise = true;
  }
  EXPECT_TRUE(saw_nonzero_noise);
}

TEST(ScaleNoiseTest, ScalesAllFourDensities) {
  ImuNoiseParams n = SampleNoise();
  const ImuNoiseParams scaled = ScaleNoise(n, 10.0);
  EXPECT_NEAR(scaled.gyro_noise_density, n.gyro_noise_density * 10.0, 1e-12);
  EXPECT_NEAR(scaled.accel_noise_density, n.accel_noise_density * 10.0, 1e-12);
  EXPECT_NEAR(scaled.gyro_random_walk, n.gyro_random_walk * 10.0, 1e-12);
  EXPECT_NEAR(scaled.accel_random_walk, n.accel_random_walk * 10.0, 1e-12);
}

}  // namespace
}  // namespace vio

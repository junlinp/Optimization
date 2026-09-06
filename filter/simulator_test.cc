#include "simulator.h"

#include "gtest/gtest.h"

namespace attitude_filter {
namespace {

TEST(Simulator, ProducesExpectedSampleCount) {
  SimulatorConfig config;
  config.duration_s = 1.0;
  config.frequency_hz = 100.0;

  const std::vector<SimulatorSample> samples = Simulate(config);
  EXPECT_EQ(samples.size(), 101u);
  EXPECT_NEAR(samples.front().t, 0.0, 1e-12);
  EXPECT_NEAR(samples.back().t, 1.0, 1e-9);
}

TEST(Simulator, TrueBodyVectorsRotateBackToTheWorldReference) {
  SimulatorConfig config;
  config.duration_s = 2.0;
  config.frequency_hz = 50.0;

  for (const SimulatorSample& s : Simulate(config)) {
    EXPECT_LT((s.R_true * s.mag_body_true - s.mag_world).norm(), 1e-9);
    EXPECT_LT((s.R_true * s.gravity_body_true - s.gravity_world).norm(), 1e-9);
  }
}

TEST(Simulator, MeasuredVectorsAreNormalized) {
  SimulatorConfig config;
  config.duration_s = 2.0;
  config.frequency_hz = 50.0;

  for (const SimulatorSample& s : Simulate(config)) {
    EXPECT_NEAR(s.mag_body_meas.norm(), 1.0, 1e-9);
    EXPECT_NEAR(s.gravity_body_meas.norm(), 1.0, 1e-9);
  }
}

TEST(Simulator, GyroBiasStaysZeroWhenDisabled) {
  SimulatorConfig config;
  config.duration_s = 5.0;
  config.frequency_hz = 100.0;
  config.use_gyro_bias = false;

  for (const SimulatorSample& s : Simulate(config)) {
    EXPECT_EQ(s.gyro_bias_true, Eigen::Vector3d::Zero());
  }
}

TEST(Simulator, GyroBiasWalksWhenEnabled) {
  SimulatorConfig config;
  config.duration_s = 30.0;
  config.frequency_hz = 100.0;
  config.gyro_bias_rw = 0.01;  // large, so it moves measurably within the run
  config.use_gyro_bias = true;

  const std::vector<SimulatorSample> samples = Simulate(config);
  EXPECT_GT(samples.back().gyro_bias_true.norm(), 1e-6);
}

}  // namespace
}  // namespace attitude_filter

#include "anees_benchmark.h"

#include <stdexcept>

#include "gtest/gtest.h"

namespace vio {
namespace {

AneesBenchmarkConfig SmallConfig() {
  AneesBenchmarkConfig config;
  config.trajectory.duration_s = 5.0;
  config.trajectory.frequency_hz = 100.0;
  config.trajectory.radius_m = 3.0;
  config.trajectory.circle_rate_rad_s = 0.3;
  config.trajectory.z_amplitude_m = 0.5;
  config.trajectory.z_rate_rad_s = 0.7;
  config.trajectory.noise.gyro_noise_density = 1.6968e-04;
  config.trajectory.noise.accel_noise_density = 2.0000e-3;
  config.trajectory.noise.gyro_random_walk = 1.9393e-05;
  config.trajectory.noise.accel_random_walk = 3.0000e-3;
  config.trajectory.seed = 7;
  config.num_monte_carlo_runs = 300;
  config.report_times_s = {1.0, 3.0, 5.0};
  return config;
}

TEST(AneesBenchmark, ThrowsWhenRunsNonPositive) {
  AneesBenchmarkConfig config = SmallConfig();
  config.num_monte_carlo_runs = 0;
  EXPECT_THROW(RunAneesBenchmark(config), std::invalid_argument);
}

TEST(AneesBenchmark, ThrowsWhenReportTimesEmpty) {
  AneesBenchmarkConfig config = SmallConfig();
  config.report_times_s.clear();
  EXPECT_THROW(RunAneesBenchmark(config), std::invalid_argument);
}

TEST(AneesBenchmark, ThrowsWhenReportTimeExceedsDuration) {
  AneesBenchmarkConfig config = SmallConfig();
  config.report_times_s = {config.trajectory.duration_s + 1.0};
  EXPECT_THROW(RunAneesBenchmark(config), std::invalid_argument);
}

TEST(AneesBenchmark, DeduplicatesAndSortsReportTimes) {
  AneesBenchmarkConfig config = SmallConfig();
  config.num_monte_carlo_runs = 5;
  config.report_times_s = {0.5, 0.1, 0.5, 0.3};

  const std::vector<AneesCurvePoint> curve = RunAneesBenchmark(config);
  ASSERT_EQ(curve.size(), 3u);
  EXPECT_LT(curve[0].t_s, curve[1].t_s);
  EXPECT_LT(curve[1].t_s, curve[2].t_s);
}

TEST(AneesBenchmark, EveryRealizationContributesWhenNoiseIsWellFormed) {
  const std::vector<AneesCurvePoint> curve = RunAneesBenchmark(SmallConfig());
  ASSERT_EQ(curve.size(), 3u);
  for (const AneesCurvePoint& pt : curve) {
    EXPECT_EQ(pt.num_samples, 300);
  }
}

// A filter whose noise parameters exactly match the simulator that generated
// the data should be statistically consistent: ANEES should sit near 1. The
// bounds are deliberately generous (Monte Carlo variance at M=300 plus
// first-order linearization error), so this is a sanity check against a
// covariance-propagation regression, not a tight statistical test.
TEST(AneesBenchmark, WellSpecifiedFilterIsApproximatelyConsistent) {
  const std::vector<AneesCurvePoint> curve = RunAneesBenchmark(SmallConfig());
  for (const AneesCurvePoint& pt : curve) {
    EXPECT_GT(pt.anees, 0.4) << "at t=" << pt.t_s;
    EXPECT_LT(pt.anees, 2.5) << "at t=" << pt.t_s;
  }
}

}  // namespace
}  // namespace vio

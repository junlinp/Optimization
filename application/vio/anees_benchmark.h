#ifndef VIO_ANEES_BENCHMARK_H_
#define VIO_ANEES_BENCHMARK_H_
#include <vector>

#include "eskf_estimator.h"
#include "imu_trajectory_simulator.h"

namespace vio {

// Configuration for a Monte Carlo consistency benchmark of EskfEstimator's
// pure IMU propagation (no measurement updates), following the ANEES
// protocol of Delama et al. 2024, "Equivariant IMU Preintegration with
// Biases: A Galilean Group Approach" (arXiv:2411.05548), Sec. V-A: for a
// fixed trajectory and noise level, run `num_monte_carlo_runs` independent
// noisy realizations and, at each requested elapsed time, report the
// Average Normalized Estimation Error Squared across all runs.
struct AneesBenchmarkConfig {
  // Trajectory shape, sample rate, gravity, and the (continuous-density)
  // noise level to test. Every realization simulated internally has fresh
  // measurement noise and a freshly drifting bias (add_measurement_noise and
  // add_bias_random_walk are forced on regardless of what's set here).
  ImuTrajectoryConfig trajectory;

  int num_monte_carlo_runs = 1000;  // M in the paper.

  // Elapsed time since the start of the run (t=0, matching the paper's
  // Delta t_ij for a preintegration window starting at t=0) at which to
  // report ANEES. Each is snapped to the nearest simulated sample; every
  // entry must lie within [0, trajectory.duration_s].
  std::vector<double> report_times_s = {0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0};
};

struct AneesCurvePoint {
  double t_s = 0.0;    // Actual (sample-snapped) time this point was computed at.
  double anees = 0.0;  // (1/(M*15)) * sum_i eps_i^T Sigma_i^-1 eps_i.
  int num_samples = 0;  // Realizations that contributed (Sigma singular -> excluded).
};

// Runs the Monte Carlo consistency benchmark and returns one AneesCurvePoint
// per distinct, time-sorted entry of config.report_times_s. A perfectly
// consistent filter reports anees == 1 at every point. Throws
// std::invalid_argument on a malformed config (non-positive runs, empty
// report_times_s, or a report time outside [0, trajectory.duration_s]).
std::vector<AneesCurvePoint> RunAneesBenchmark(const AneesBenchmarkConfig& config);

}  // namespace vio
#endif  // VIO_ANEES_BENCHMARK_H_

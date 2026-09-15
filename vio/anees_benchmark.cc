#include "anees_benchmark.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace vio {
namespace {

// Error state ordered [dp, dv, dtheta, db_g, db_a], matching EskfState's
// documented convention: p_true=p+dp, v_true=v+dv, R_true=R*Exp(dtheta),
// b_g_true=b_g+db_g, b_a_true=b_a+db_a.
Eigen::Matrix<double, 15, 1> ErrorState(const EskfState& est, const ImuTrajectorySample& truth) {
  Eigen::Matrix<double, 15, 1> eps;
  eps.segment<3>(0) = truth.p_true - est.p;
  eps.segment<3>(3) = truth.v_true - est.v;
  eps.segment<3>(6) = (est.R.inverse() * truth.R_true).log();
  eps.segment<3>(9) = truth.gyro_bias_true - est.bias_gyro;
  eps.segment<3>(12) = truth.accel_bias_true - est.bias_accel;
  return eps;
}

}  // namespace

std::vector<AneesCurvePoint> RunAneesBenchmark(const AneesBenchmarkConfig& config) {
  if (config.num_monte_carlo_runs <= 0) {
    throw std::invalid_argument("num_monte_carlo_runs must be positive");
  }
  if (config.report_times_s.empty()) {
    throw std::invalid_argument("report_times_s must not be empty");
  }
  if (config.trajectory.frequency_hz <= 0.0 || config.trajectory.duration_s <= 0.0) {
    throw std::invalid_argument("trajectory duration and frequency must be positive");
  }

  const double dt = 1.0 / config.trajectory.frequency_hz;

  std::vector<std::size_t> report_indices;
  report_indices.reserve(config.report_times_s.size());
  for (const double t : config.report_times_s) {
    if (t < 0.0 || t > config.trajectory.duration_s + 1e-9) {
      throw std::invalid_argument("report time out of [0, trajectory.duration_s] range");
    }
    report_indices.push_back(static_cast<std::size_t>(std::llround(t / dt)));
  }
  std::sort(report_indices.begin(), report_indices.end());
  report_indices.erase(std::unique(report_indices.begin(), report_indices.end()),
                        report_indices.end());

  std::vector<double> sum_nees(report_indices.size(), 0.0);
  std::vector<int> count(report_indices.size(), 0);

  // A single seed sequence derives one fresh, independent seed per Monte
  // Carlo realization from config.trajectory.seed, so the whole run is
  // reproducible from that one number.
  std::mt19937 seed_rng(config.trajectory.seed);

  for (int run = 0; run < config.num_monte_carlo_runs; ++run) {
    ImuTrajectoryConfig traj_cfg = config.trajectory;
    traj_cfg.seed = seed_rng();
    traj_cfg.add_measurement_noise = true;
    traj_cfg.add_bias_random_walk = true;

    const std::vector<ImuTrajectorySample> samples = SimulateImuTrajectory(traj_cfg);

    // The filter starts exactly at the trajectory's true initial pose (zero
    // bias estimate), matching the paper's preintegration starting from a
    // known reference (Upsilon_0 = I): this isolates propagation
    // consistency from any initialization error.
    EskfState x0;
    x0.p = samples.front().p_true;
    x0.v = samples.front().v_true;
    x0.R = samples.front().R_true;

    EskfEstimator est(x0, Eigen::Matrix<double, 15, 15>::Zero(), traj_cfg.noise,
                       traj_cfg.gravity_world);

    std::size_t next_report = 0;
    for (std::size_t k = 0; k < samples.size() && next_report < report_indices.size(); ++k) {
      if (k > 0) est.Predict(samples[k].gyro_meas, samples[k].accel_meas, dt);

      if (report_indices[next_report] == k) {
        const Eigen::Matrix<double, 15, 1> eps = ErrorState(est.state(), samples[k]);
        const Eigen::LDLT<Eigen::Matrix<double, 15, 15>> ldlt(est.covariance());
        // Sigma is exactly zero at k=0 (no process noise has accumulated
        // yet) and hence singular; skip rather than divide by zero. Any
        // later singularity would indicate a genuine filter bug, but is
        // handled the same way so one bad realization can't crash the run.
        if (ldlt.info() == Eigen::Success) {
          sum_nees[next_report] += eps.dot(ldlt.solve(eps));
          count[next_report] += 1;
        }
        ++next_report;
      }
    }
  }

  std::vector<AneesCurvePoint> curve;
  curve.reserve(report_indices.size());
  for (std::size_t j = 0; j < report_indices.size(); ++j) {
    AneesCurvePoint pt;
    pt.t_s = static_cast<double>(report_indices[j]) * dt;
    pt.num_samples = count[j];
    pt.anees = count[j] > 0 ? sum_nees[j] / (static_cast<double>(count[j]) * 15.0) : 0.0;
    curve.push_back(pt);
  }
  return curve;
}

}  // namespace vio

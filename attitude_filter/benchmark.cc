// Usage: benchmark [--duration=S] [--frequency=Hz] [--seed=N] [--gyro-bias]
//                  [--format=text|markdown]
//
// Runs the attitude ESKF over a simulated tumbling-rigid-body sequence (see
// simulator.h, ported from the user-supplied attitude_simulator.cpp) and
// reports its orientation error against the simulator's ground truth. The
// filter starts from a deliberately wrong orientation guess (identity, with
// a large initial covariance) so the report separates the startup
// convergence transient from steady-state tracking error.
//
// --format=text (default): human-readable report on stdout.
// --format=markdown: a ready-to-post Markdown table on stdout.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "eskf_attitude.h"
#include "simulator.h"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSettleTimeS = 3.0;  // excluded from "steady-state" stats

struct ErrorStats {
  int num_samples = 0;
  double rmse_deg = 0;
  double mean_deg = 0;
  double median_deg = 0;
  double max_deg = 0;
};

ErrorStats ComputeStats(std::vector<double> errors_deg) {
  ErrorStats stats;
  stats.num_samples = static_cast<int>(errors_deg.size());
  if (errors_deg.empty()) return stats;

  double sum = 0.0, sum_sq = 0.0;
  for (const double e : errors_deg) {
    sum += e;
    sum_sq += e * e;
    stats.max_deg = std::max(stats.max_deg, e);
  }
  stats.mean_deg = sum / errors_deg.size();
  stats.rmse_deg = std::sqrt(sum_sq / errors_deg.size());

  std::sort(errors_deg.begin(), errors_deg.end());
  stats.median_deg = errors_deg[errors_deg.size() / 2];
  return stats;
}

void PrintUsage(const char* argv0) {
  std::cerr << "Usage: " << argv0
           << " [--duration=S] [--frequency=Hz] [--seed=N] [--gyro-bias] "
              "[--format=text|markdown]\n";
}

std::string FormatDeg(double value) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4) << value;
  return oss.str();
}

void PrintText(const attitude_filter::SimulatorConfig& config, const ErrorStats& overall,
              const ErrorStats& steady_state) {
  std::cout << "Attitude ESKF benchmark\n"
            << "  duration: " << config.duration_s << " s, frequency: " << config.frequency_hz
            << " Hz, gyro bias random walk: " << (config.use_gyro_bias ? "on" : "off") << "\n\n"
            << "Overall (includes startup convergence, " << overall.num_samples << " samples)\n"
            << "  RMSE:   " << FormatDeg(overall.rmse_deg) << " deg\n"
            << "  mean:   " << FormatDeg(overall.mean_deg) << " deg\n"
            << "  median: " << FormatDeg(overall.median_deg) << " deg\n"
            << "  max:    " << FormatDeg(overall.max_deg) << " deg\n\n"
            << "Steady-state (t >= " << kSettleTimeS << "s, " << steady_state.num_samples
            << " samples)\n"
            << "  RMSE:   " << FormatDeg(steady_state.rmse_deg) << " deg\n"
            << "  mean:   " << FormatDeg(steady_state.mean_deg) << " deg\n"
            << "  median: " << FormatDeg(steady_state.median_deg) << " deg\n"
            << "  max:    " << FormatDeg(steady_state.max_deg) << " deg\n";
}

void PrintMarkdown(const attitude_filter::SimulatorConfig& config, const ErrorStats& overall,
                   const ErrorStats& steady_state) {
  std::cout << "### Attitude ESKF benchmark\n\n"
            << "duration=" << config.duration_s << "s, frequency=" << config.frequency_hz
            << "Hz, gyro bias random walk=" << (config.use_gyro_bias ? "on" : "off") << "\n\n"
            << "| Metric | Overall | Steady-state (t >= " << kSettleTimeS << "s) |\n"
            << "|---|---|---|\n"
            << "| RMSE (deg) | " << FormatDeg(overall.rmse_deg) << " | "
            << FormatDeg(steady_state.rmse_deg) << " |\n"
            << "| Mean (deg) | " << FormatDeg(overall.mean_deg) << " | "
            << FormatDeg(steady_state.mean_deg) << " |\n"
            << "| Median (deg) | " << FormatDeg(overall.median_deg) << " | "
            << FormatDeg(steady_state.median_deg) << " |\n"
            << "| Max (deg) | " << FormatDeg(overall.max_deg) << " | "
            << FormatDeg(steady_state.max_deg) << " |\n"
            << "| Samples | " << overall.num_samples << " | " << steady_state.num_samples
            << " |\n";
}

}  // namespace

int main(int argc, char** argv) {
  attitude_filter::SimulatorConfig config;
  std::string format = "text";

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("--format=", 0) == 0) {
      format = arg.substr(std::string("--format=").size());
    } else if (arg.rfind("--duration=", 0) == 0) {
      config.duration_s = std::stod(arg.substr(std::string("--duration=").size()));
    } else if (arg.rfind("--frequency=", 0) == 0) {
      config.frequency_hz = std::stod(arg.substr(std::string("--frequency=").size()));
    } else if (arg.rfind("--seed=", 0) == 0) {
      config.seed =
          static_cast<std::uint32_t>(std::stoul(arg.substr(std::string("--seed=").size())));
    } else if (arg == "--gyro-bias") {
      config.use_gyro_bias = true;
    } else {
      PrintUsage(argv[0]);
      return 1;
    }
  }

  if (format != "text" && format != "markdown") {
    PrintUsage(argv[0]);
    return 1;
  }

  std::vector<attitude_filter::SimulatorSample> samples;
  try {
    samples = attitude_filter::Simulate(config);
  } catch (const std::exception& e) {
    std::cerr << "Simulation failed: " << e.what() << "\n";
    return 1;
  }
  if (samples.empty()) {
    std::cerr << "Simulator produced no samples.\n";
    return 1;
  }

  // The simulator's gyro_noise_std is a fixed per-sample additive std, not a
  // continuous-time density: its equivalent density is std / sqrt(dt), so
  // that (density^2 * dt) reproduces the same per-sample variance the
  // simulator actually injected. Getting this wrong silently mistunes the
  // filter (too little or too much trust in gyro integration) without
  // making it obviously fail, so it's worth calling out explicitly.
  const double dt = 1.0 / config.frequency_hz;
  attitude_filter::AttitudeImuNoiseParams noise;
  noise.gyro_noise_density = config.gyro_noise_std / std::sqrt(dt);
  noise.gyro_random_walk = config.gyro_bias_rw;

  attitude_filter::AttitudeEskfState x0;  // deliberately wrong: identity, not the true attitude
  Eigen::Matrix<double, 6, 6> P0 = Eigen::Matrix<double, 6, 6>::Zero();
  const double initial_orientation_std = 30.0 * kPi / 180.0;
  const double initial_bias_std = 0.05;
  P0.block<3, 3>(0, 0) =
      (initial_orientation_std * initial_orientation_std) * Eigen::Matrix3d::Identity();
  P0.block<3, 3>(3, 3) = (initial_bias_std * initial_bias_std) * Eigen::Matrix3d::Identity();

  attitude_filter::AttitudeEskf est(x0, P0, noise);

  std::vector<double> overall_errors_deg;
  std::vector<double> steady_state_errors_deg;
  overall_errors_deg.reserve(samples.size());

  for (std::size_t k = 0; k < samples.size(); ++k) {
    const attitude_filter::SimulatorSample& s = samples[k];
    if (k > 0) est.Predict(s.omega_meas, dt);

    attitude_filter::VectorMeasurement gravity;
    gravity.world_reference = s.gravity_world;
    gravity.body_measured = s.gravity_body_meas;
    gravity.sigma = config.gravity_noise_std;
    est.UpdateVector(gravity);

    attitude_filter::VectorMeasurement mag;
    mag.world_reference = s.mag_world;
    mag.body_measured = s.mag_body_meas;
    mag.sigma = config.magnetometer_noise_std;
    est.UpdateVector(mag);

    const double error_deg =
        (est.state().R.inverse() * s.R_true).log().norm() * 180.0 / kPi;
    overall_errors_deg.push_back(error_deg);
    if (s.t >= kSettleTimeS) steady_state_errors_deg.push_back(error_deg);
  }

  const ErrorStats overall = ComputeStats(overall_errors_deg);
  const ErrorStats steady_state = ComputeStats(steady_state_errors_deg);

  if (format == "markdown") {
    PrintMarkdown(config, overall, steady_state);
  } else {
    PrintText(config, overall, steady_state);
  }

  return 0;
}

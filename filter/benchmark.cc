// Usage: benchmark [--duration=S] [--frequency=Hz] [--seed=N] [--gyro-bias]
//                  [--format=text|markdown]
//
// Runs the attitude ESKF, EqF, and EqF* over the same simulated tumbling-rigid-body
// sequence (see simulator.h) and reports each filter's orientation error
// against the simulator's ground truth. Both filters start from the same
// deliberately wrong orientation guess (identity, with a large initial
// covariance) so the report separates startup convergence from steady-state
// tracking error.
//
// --format=text (default): human-readable report on stdout.
// --format=markdown: a ready-to-post Markdown table on stdout.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "eqf_attitude.h"
#include "eqf_star_attitude.h"
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

struct BenchmarkResult {
  ErrorStats overall;
  ErrorStats steady_state;
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

Eigen::Matrix<double, 6, 6> MakeInitialCovariance() {
  Eigen::Matrix<double, 6, 6> P0 = Eigen::Matrix<double, 6, 6>::Zero();
  const double initial_orientation_std = 30.0 * kPi / 180.0;
  const double initial_bias_std = 0.05;
  P0.block<3, 3>(0, 0) =
      (initial_orientation_std * initial_orientation_std) * Eigen::Matrix3d::Identity();
  P0.block<3, 3>(3, 3) = (initial_bias_std * initial_bias_std) * Eigen::Matrix3d::Identity();
  return P0;
}

template <typename StateAccessor>
BenchmarkResult RunFilter(
    const std::vector<attitude_filter::SimulatorSample>& samples, double dt,
    const std::function<void(const attitude_filter::SimulatorSample&, double)>& step,
    StateAccessor state) {
  std::vector<double> overall_errors_deg;
  std::vector<double> steady_state_errors_deg;
  overall_errors_deg.reserve(samples.size());

  for (std::size_t k = 0; k < samples.size(); ++k) {
    const attitude_filter::SimulatorSample& s = samples[k];
    step(s, k == 0 ? 0.0 : dt);

    const double error_deg =
        (state().R.inverse() * s.R_true).log().norm() * 180.0 / kPi;
    overall_errors_deg.push_back(error_deg);
    if (s.t >= kSettleTimeS) steady_state_errors_deg.push_back(error_deg);
  }

  BenchmarkResult result;
  result.overall = ComputeStats(overall_errors_deg);
  result.steady_state = ComputeStats(steady_state_errors_deg);
  return result;
}

BenchmarkResult RunEskf(
    const std::vector<attitude_filter::SimulatorSample>& samples,
    const attitude_filter::SimulatorConfig& config, double dt) {
  attitude_filter::AttitudeImuNoiseParams noise;
  noise.gyro_noise_density = config.gyro_noise_std / std::sqrt(dt);
  noise.gyro_random_walk = config.gyro_bias_rw;

  attitude_filter::AttitudeEskfState x0;
  attitude_filter::AttitudeEskf est(x0, MakeInitialCovariance(), noise);

  return RunFilter(
      samples, dt,
      [&](const attitude_filter::SimulatorSample& s, double predict_dt) {
        if (predict_dt > 0.0) est.Predict(s.omega_meas, predict_dt);

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
      },
      [&est]() -> const attitude_filter::AttitudeEskfState& { return est.state(); });
}

BenchmarkResult RunEqf(
    const std::vector<attitude_filter::SimulatorSample>& samples,
    const attitude_filter::SimulatorConfig& config, double dt) {
  eqf_attitude::EqfImuNoiseParams noise;
  noise.gyro_noise_density = config.gyro_noise_std / std::sqrt(dt);
  noise.gyro_random_walk = config.gyro_bias_rw;

  eqf_attitude::EqfAttitudeState x0;
  Eigen::Matrix<double, 3, 3> P0 = MakeInitialCovariance().block<3, 3>(0, 0);
  eqf_attitude::EqfAttitude est(x0, P0, noise, samples.front().gravity_world,
                                samples.front().mag_world);

  return RunFilter(
      samples, dt,
      [&](const attitude_filter::SimulatorSample& s, double predict_dt) {
        if (predict_dt > 0.0) est.Predict(s.omega_meas, predict_dt);

        eqf_attitude::VectorMeasurement gravity;
        gravity.body_measured = s.gravity_body_meas;
        gravity.sigma = config.gravity_noise_std;

        eqf_attitude::VectorMeasurement mag;
        mag.body_measured = s.mag_body_meas;
        mag.sigma = config.magnetometer_noise_std;
        est.UpdateVector(gravity, mag, dt);
      },
      [&est]() -> const eqf_attitude::EqfAttitudeState& { return est.state(); });
}

BenchmarkResult RunEqfStar(
    const std::vector<attitude_filter::SimulatorSample>& samples,
    const attitude_filter::SimulatorConfig& config, double dt) {
  eqf_star_attitude::EqfStarImuNoiseParams noise;
  noise.gyro_noise_density = config.gyro_noise_std / std::sqrt(dt);
  noise.gyro_random_walk = config.gyro_bias_rw;

  eqf_star_attitude::EqfStarAttitudeState x0;
  Eigen::Matrix<double, 3, 3> P0 = MakeInitialCovariance().block<3, 3>(0, 0);
  eqf_star_attitude::EqfStarAttitude est(x0, P0, noise, samples.front().gravity_world,
                                       samples.front().mag_world);

  return RunFilter(
      samples, dt,
      [&](const attitude_filter::SimulatorSample& s, double predict_dt) {
        if (predict_dt > 0.0) est.Predict(s.omega_meas, predict_dt);

        eqf_star_attitude::VectorMeasurement gravity;
        gravity.body_measured = s.gravity_body_meas;
        gravity.sigma = config.gravity_noise_std;

        eqf_star_attitude::VectorMeasurement mag;
        mag.body_measured = s.mag_body_meas;
        mag.sigma = config.magnetometer_noise_std;
        est.UpdateVector(gravity, mag, dt);
      },
      [&est]() -> const eqf_star_attitude::EqfStarAttitudeState& { return est.state(); });
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

void PrintText(const attitude_filter::SimulatorConfig& config,
               const BenchmarkResult& eskf, const BenchmarkResult& eqf,
               const BenchmarkResult& eqf_star) {
  std::cout << "Attitude filter benchmark (ESKF vs EqF vs EqF*)\n"
            << "  duration: " << config.duration_s << " s, frequency: "
            << config.frequency_hz << " Hz, gyro bias random walk: "
            << (config.use_gyro_bias ? "on" : "off") << "\n\n";

  const auto print_stats = [](const char* label, const ErrorStats& eskf_stats,
                              const ErrorStats& eqf_stats,
                              const ErrorStats& eqf_star_stats) {
    std::cout << label << "\n"
              << "  ESKF RMSE:    " << FormatDeg(eskf_stats.rmse_deg) << " deg\n"
              << "  EqF  RMSE:    " << FormatDeg(eqf_stats.rmse_deg) << " deg\n"
              << "  EqF* RMSE:    " << FormatDeg(eqf_star_stats.rmse_deg) << " deg\n"
              << "  ESKF mean:    " << FormatDeg(eskf_stats.mean_deg) << " deg\n"
              << "  EqF  mean:    " << FormatDeg(eqf_stats.mean_deg) << " deg\n"
              << "  EqF* mean:    " << FormatDeg(eqf_star_stats.mean_deg) << " deg\n"
              << "  ESKF median:  " << FormatDeg(eskf_stats.median_deg) << " deg\n"
              << "  EqF  median:  " << FormatDeg(eqf_stats.median_deg) << " deg\n"
              << "  EqF* median:  " << FormatDeg(eqf_star_stats.median_deg) << " deg\n"
              << "  ESKF max:     " << FormatDeg(eskf_stats.max_deg) << " deg\n"
              << "  EqF  max:     " << FormatDeg(eqf_stats.max_deg) << " deg\n"
              << "  EqF* max:     " << FormatDeg(eqf_star_stats.max_deg) << " deg\n"
              << "  samples:      " << eskf_stats.num_samples << "\n\n";
  };

  print_stats(("Overall (includes startup convergence)"), eskf.overall, eqf.overall,
              eqf_star.overall);
  print_stats(("Steady-state (t >= " + std::to_string(kSettleTimeS) + " s)").c_str(),
              eskf.steady_state, eqf.steady_state, eqf_star.steady_state);
}

void PrintMarkdown(const attitude_filter::SimulatorConfig& config,
                   const BenchmarkResult& eskf, const BenchmarkResult& eqf,
                   const BenchmarkResult& eqf_star) {
  std::cout << "### Attitude filter benchmark (ESKF vs EqF vs EqF*)\n\n"
            << "duration=" << config.duration_s << "s, frequency=" << config.frequency_hz
            << "Hz, gyro bias random walk=" << (config.use_gyro_bias ? "on" : "off") << "\n\n"
            << "| Metric | ESKF overall | EqF overall | EqF* overall | "
               "ESKF steady | EqF steady | EqF* steady |\n"
            << "|---|---|---|---|---|---|---|\n"
            << "| RMSE (deg) | " << FormatDeg(eskf.overall.rmse_deg) << " | "
            << FormatDeg(eqf.overall.rmse_deg) << " | "
            << FormatDeg(eqf_star.overall.rmse_deg) << " | "
            << FormatDeg(eskf.steady_state.rmse_deg) << " | "
            << FormatDeg(eqf.steady_state.rmse_deg) << " | "
            << FormatDeg(eqf_star.steady_state.rmse_deg) << " |\n"
            << "| Mean (deg) | " << FormatDeg(eskf.overall.mean_deg) << " | "
            << FormatDeg(eqf.overall.mean_deg) << " | "
            << FormatDeg(eqf_star.overall.mean_deg) << " | "
            << FormatDeg(eskf.steady_state.mean_deg) << " | "
            << FormatDeg(eqf.steady_state.mean_deg) << " | "
            << FormatDeg(eqf_star.steady_state.mean_deg) << " |\n"
            << "| Median (deg) | " << FormatDeg(eskf.overall.median_deg) << " | "
            << FormatDeg(eqf.overall.median_deg) << " | "
            << FormatDeg(eqf_star.overall.median_deg) << " | "
            << FormatDeg(eskf.steady_state.median_deg) << " | "
            << FormatDeg(eqf.steady_state.median_deg) << " | "
            << FormatDeg(eqf_star.steady_state.median_deg) << " |\n"
            << "| Max (deg) | " << FormatDeg(eskf.overall.max_deg) << " | "
            << FormatDeg(eqf.overall.max_deg) << " | "
            << FormatDeg(eqf_star.overall.max_deg) << " | "
            << FormatDeg(eskf.steady_state.max_deg) << " | "
            << FormatDeg(eqf.steady_state.max_deg) << " | "
            << FormatDeg(eqf_star.steady_state.max_deg) << " |\n"
            << "| Samples | " << eskf.overall.num_samples << " | "
            << eqf.overall.num_samples << " | " << eqf_star.overall.num_samples << " | "
            << eskf.steady_state.num_samples << " | " << eqf.steady_state.num_samples
            << " | " << eqf_star.steady_state.num_samples << " |\n";
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

  const double dt = 1.0 / config.frequency_hz;
  const BenchmarkResult eskf = RunEskf(samples, config, dt);
  const BenchmarkResult eqf = RunEqf(samples, config, dt);
  const BenchmarkResult eqf_star = RunEqfStar(samples, config, dt);

  if (format == "markdown") {
    PrintMarkdown(config, eskf, eqf, eqf_star);
  } else {
    PrintText(config, eskf, eqf, eqf_star);
  }

  return 0;
}

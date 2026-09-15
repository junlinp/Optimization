// Usage: imu_mc_benchmark [--runs=M] [--duration=S] [--frequency=Hz]
//                         [--seed=N] [--format=text|markdown]
//
// Monte Carlo ANEES consistency benchmark for EskfEstimator's pure IMU
// propagation (no measurement updates), reproducing the protocol in Sec.
// V-A of Delama et al., "Equivariant IMU Preintegration with Biases: A
// Galilean Group Approach" (arXiv:2411.05548): M independent noisy
// realizations of a circular-plus-cosine trajectory (see
// vio/imu_trajectory_simulator.h) are run through the filter's own
// covariance propagation, and ANEES is reported at a set of elapsed times
// for three noise levels (low/medium/high, lambda = 0.1/1/10 scaling the
// paper's baseline discrete sigmas). A perfectly consistent filter reports
// ANEES == 1 everywhere.
//
// The paper does not publish the exact trajectory radius/rate it used, so
// this reproduces the qualitative pattern, not the exact numeric curve --
// see the reconstruction note in vio/imu_trajectory_simulator.h.
//
// --runs defaults to 200 (a few seconds); pass --runs=1000 to match the
// paper's M exactly (several minutes at the default 30s/200Hz settings).
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "anees_benchmark.h"
#include "imu_trajectory_simulator.h"

namespace {

constexpr double kPaperFrequencyHz = 200.0;
// Baseline (lambda=1) discrete-time sigmas from Delama et al. 2024, Fig. 1/2
// captions, at their 200 Hz sample rate.
constexpr double kPaperSigmaDGyro = 7e-2;            // rad/s
constexpr double kPaperSigmaDAccel = 1.9e-1;         // m/s^2
constexpr double kPaperSigmaDGyroBiasRw = 1.5e-4;    // rad/s^2
constexpr double kPaperSigmaDAccelBiasRw = 1.2e-2;   // m/s^3

// Converts the paper's discrete sigma (sigma_d = sigma_c / sqrt(dt)) into
// the continuous-time density ImuTrajectoryConfig::noise expects.
vio::ImuNoiseParams PaperBaselineNoise(double frequency_hz) {
  const double sqrt_dt = std::sqrt(1.0 / frequency_hz);
  vio::ImuNoiseParams noise;
  noise.gyro_noise_density = kPaperSigmaDGyro * sqrt_dt;
  noise.accel_noise_density = kPaperSigmaDAccel * sqrt_dt;
  noise.gyro_random_walk = kPaperSigmaDGyroBiasRw * sqrt_dt;
  noise.accel_random_walk = kPaperSigmaDAccelBiasRw * sqrt_dt;
  return noise;
}

struct NoiseLevel {
  const char* label;
  double lambda;
};

constexpr NoiseLevel kNoiseLevels[] = {
    {"low (lambda=0.1)", 0.1},
    {"medium (lambda=1)", 1.0},
    {"high (lambda=10)", 10.0},
};
constexpr std::size_t kNumNoiseLevels = sizeof(kNoiseLevels) / sizeof(kNoiseLevels[0]);

std::vector<double> BuildReportTimes(double duration_s) {
  const std::vector<double> candidates = {0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0};
  std::vector<double> report_times_s;
  for (const double t : candidates) {
    if (t <= duration_s + 1e-9) report_times_s.push_back(t);
  }
  if (report_times_s.empty() || report_times_s.back() < duration_s - 1e-9) {
    report_times_s.push_back(duration_s);
  }
  return report_times_s;
}

void PrintUsage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " [--runs=M] [--duration=S] [--frequency=Hz] [--seed=N] "
               "[--format=text|markdown]\n";
}

std::string FormatAnees(double value) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3) << value;
  return oss.str();
}

void PrintText(const std::vector<std::vector<vio::AneesCurvePoint>>& curves) {
  std::cout << "IMU Monte Carlo ANEES consistency benchmark (EskfEstimator propagation-only)\n"
            << "A perfectly consistent filter reports ANEES == 1 at every point.\n\n";
  for (std::size_t level = 0; level < curves.size(); ++level) {
    std::cout << kNoiseLevels[level].label << ":\n";
    for (const vio::AneesCurvePoint& pt : curves[level]) {
      std::cout << "  t=" << std::fixed << std::setprecision(1) << pt.t_s
                << "s  ANEES=" << FormatAnees(pt.anees) << "  (M=" << pt.num_samples << ")\n";
    }
    std::cout << "\n";
  }
}

void PrintMarkdown(const std::vector<double>& report_times_s,
                   const std::vector<std::vector<vio::AneesCurvePoint>>& curves) {
  std::cout << "### IMU Monte Carlo ANEES consistency benchmark\n\n| Noise level |";
  for (const double t : report_times_s) std::cout << " t=" << t << "s |";
  std::cout << "\n|---|";
  for (std::size_t i = 0; i < report_times_s.size(); ++i) std::cout << "---|";
  std::cout << "\n";
  for (std::size_t level = 0; level < curves.size(); ++level) {
    std::cout << "| " << kNoiseLevels[level].label << " |";
    for (const vio::AneesCurvePoint& pt : curves[level]) {
      std::cout << " " << FormatAnees(pt.anees) << " |";
    }
    std::cout << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  int num_runs = 200;
  double duration_s = 30.0;
  double frequency_hz = kPaperFrequencyHz;
  std::uint32_t seed = 42;
  std::string format = "text";

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("--runs=", 0) == 0) {
      num_runs = std::stoi(arg.substr(std::string("--runs=").size()));
    } else if (arg.rfind("--duration=", 0) == 0) {
      duration_s = std::stod(arg.substr(std::string("--duration=").size()));
    } else if (arg.rfind("--frequency=", 0) == 0) {
      frequency_hz = std::stod(arg.substr(std::string("--frequency=").size()));
    } else if (arg.rfind("--seed=", 0) == 0) {
      seed = static_cast<std::uint32_t>(std::stoul(arg.substr(std::string("--seed=").size())));
    } else if (arg.rfind("--format=", 0) == 0) {
      format = arg.substr(std::string("--format=").size());
    } else {
      PrintUsage(argv[0]);
      return 1;
    }
  }
  if (format != "text" && format != "markdown") {
    PrintUsage(argv[0]);
    return 1;
  }

  vio::ImuTrajectoryConfig trajectory;
  trajectory.duration_s = duration_s;
  trajectory.frequency_hz = frequency_hz;
  trajectory.radius_m = 5.0;
  // radius * circle_rate = 0.9 m/s, matching the paper's example average speed.
  trajectory.circle_rate_rad_s = 0.18;
  trajectory.seed = seed;
  trajectory.noise = PaperBaselineNoise(frequency_hz);

  const std::vector<double> report_times_s = BuildReportTimes(duration_s);

  std::vector<std::vector<vio::AneesCurvePoint>> curves;
  curves.reserve(kNumNoiseLevels);
  try {
    for (const NoiseLevel& level : kNoiseLevels) {
      vio::AneesBenchmarkConfig config;
      config.trajectory = trajectory;
      config.trajectory.noise = vio::ScaleNoise(trajectory.noise, level.lambda);
      config.num_monte_carlo_runs = num_runs;
      config.report_times_s = report_times_s;
      curves.push_back(vio::RunAneesBenchmark(config));
    }
  } catch (const std::exception& e) {
    std::cerr << "Benchmark failed: " << e.what() << "\n";
    return 1;
  }

  if (format == "markdown") {
    PrintMarkdown(report_times_s, curves);
  } else {
    PrintText(curves);
  }
  return 0;
}

// Usage: eqvio_sim_benchmark [--runs=M] [--duration=S] [--seed=N]
//                            [--init-scale=K] [--format=text|markdown]
//
// Monte Carlo consistency and convergence benchmark for EqvioEstimator on the
// synthetic scene of vio/vio_simulator.h (the extrinsic/bias convergence
// experiment of van Goor & Mahony, EqVIO, arXiv:2205.01980, Sec. 7.3). Each
// run draws a fresh landmark layout, fresh IMU and pixel noise, true IMU
// biases from the filter's bias prior, and an initial estimate whose error is
// drawn from the filter's own initial covariance -- so a consistent filter
// reports an average NEES / dim of 1 (Sec. 7.2's metric
// NEES = eps^T Sigma^-1 eps, eps = vartheta(phi(X_hat^-1, xi))).
// --init-scale multiplies every initial-error standard deviation.
//
// Build with -c opt: `bazelisk run -c opt //vio:eqvio_sim_benchmark`.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "eqvio.h"
#include "eskf_estimator.h"
#include "vio_simulator.h"

namespace {

using Eigen::Vector3d;

// EuRoC/ADIS16448-class IMU, as in the paper's Table 2 simulations.
constexpr double kGyroNoise = 1.7e-4, kAccelNoise = 2.0e-3;
constexpr double kGyroWalk = 1.9e-5, kAccelWalk = 3.0e-3;

// Initial-error standard deviations. The extrinsic values are the paper's
// Sec. 7.3 choice; the rest are a moderate start-up uncertainty.
double kSigmaAttitude = 0.02, kSigmaPosition = 0.01, kSigmaVelocity = 0.05;
double kSigmaGyroBias = 0.02, kSigmaAccelBias = 0.1;
double kSigmaExtrinsicRot = 0.05, kSigmaExtrinsicTrans = 0.05;

struct Options {
  int runs = 10;
  double duration_s = 30.0;
  std::uint32_t seed = 1;
  double init_scale = 1.0;
  bool markdown = false;
};

struct Accumulator {
  double t = 0;
  double pos_sq = 0, att_sq = 0;
  double nees_full = 0, nees_pose = 0, nees_att = 0;
  int count = 0;
};

Vector3d Draw(std::mt19937& rng, double sigma) {
  std::normal_distribution<double> n(0.0, sigma);
  return {n(rng), n(rng), n(rng)};
}

double Nees(const Eigen::VectorXd& eps, const Eigen::MatrixXd& sigma) {
  return eps.dot(sigma.ldlt().solve(eps));
}

bool ParseArgs(int argc, char** argv, Options* opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const char* prefix) -> const char* {
      const std::string p = prefix;
      return arg.rfind(p, 0) == 0 ? argv[i] + p.size() : nullptr;
    };
    if (const char* v = value("--runs=")) {
      opt->runs = std::atoi(v);
    } else if (const char* v = value("--duration=")) {
      opt->duration_s = std::atof(v);
    } else if (const char* v = value("--seed=")) {
      opt->seed = static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
    } else if (const char* v = value("--init-scale=")) {
      opt->init_scale = std::atof(v);
    } else if (const char* v = value("--format=")) {
      opt->markdown = std::string(v) == "markdown";
    } else {
      return false;
    }
  }
  return opt->runs > 0 && opt->duration_s > 0 && opt->init_scale > 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!ParseArgs(argc, argv, &opt)) {
    std::cerr << "Usage: " << argv[0]
              << " [--runs=M] [--duration=S] [--seed=N] [--init-scale=K] [--format=text|markdown]\n";
    return 2;
  }
  for (double* sigma : {&kSigmaAttitude, &kSigmaPosition, &kSigmaVelocity, &kSigmaGyroBias,
                        &kSigmaAccelBias, &kSigmaExtrinsicRot, &kSigmaExtrinsicTrans}) {
    *sigma *= opt.init_scale;
  }

  const std::vector<double> report_times = [&] {
    std::vector<double> out;
    for (const double t : {1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 90.0}) {
      if (t <= opt.duration_s + 1e-9) out.push_back(t);
    }
    return out;
  }();
  std::vector<Accumulator> acc(report_times.size());
  double final_gyro_bias_sq = 0, final_accel_bias_sq = 0;
  double final_ext_rot_sq = 0, final_ext_trans_sq = 0, dr_final_sq = 0;
  double tracked_sum = 0;
  int tracked_count = 0;

  for (int run = 0; run < opt.runs; ++run) {
    std::mt19937 rng(opt.seed * 7919u + static_cast<std::uint32_t>(run));

    vio::VioSimConfig config;
    config.duration_s = opt.duration_s;
    config.noise.gyro_noise_density = kGyroNoise;
    config.noise.accel_noise_density = kAccelNoise;
    config.noise.gyro_random_walk = kGyroWalk;
    config.noise.accel_random_walk = kAccelWalk;
    config.add_bias_random_walk = true;
    config.initial_bias_gyro = Draw(rng, kSigmaGyroBias);
    config.initial_bias_accel = Draw(rng, kSigmaAccelBias);
    config.seed = rng();
    const vio::VioSimulation sim = vio::SimulateVio(config);

    // Initial estimate: N_hat = exp(eps_N)^-1 N, b_hat = 0 (true biases were
    // drawn from the prior), T_hat = T Exp(-tau).
    const vio::EqvioState truth0 = vio::TruthState(sim, 0);
    vio::Vector9d eps_nav;
    eps_nav << Draw(rng, kSigmaAttitude), Draw(rng, kSigmaPosition), Draw(rng, kSigmaVelocity);
    const vio::SE23 N_hat =
        vio::SE23::exp(eps_nav).inverse() * vio::SE23{truth0.R, truth0.p, truth0.v};
    vio::Vector6d tau;
    tau << Draw(rng, kSigmaExtrinsicTrans), Draw(rng, kSigmaExtrinsicRot);
    vio::EqvioState init;
    init.R = N_hat.R;
    init.p = N_hat.x;
    init.v = N_hat.v;
    init.T_body_camera = truth0.T_body_camera * Sophus::SE3d::exp(-tau);

    Eigen::Matrix<double, 9, 9> nav = Eigen::Matrix<double, 9, 9>::Zero();
    nav.diagonal() << Vector3d::Constant(kSigmaAttitude * kSigmaAttitude),
        Vector3d::Constant(kSigmaPosition * kSigmaPosition),
        Vector3d::Constant(kSigmaVelocity * kSigmaVelocity);
    Eigen::Matrix<double, 6, 6> bias = Eigen::Matrix<double, 6, 6>::Zero();
    bias.diagonal() << Vector3d::Constant(kSigmaGyroBias * kSigmaGyroBias),
        Vector3d::Constant(kSigmaAccelBias * kSigmaAccelBias);
    Eigen::Matrix<double, 6, 6> ext = Eigen::Matrix<double, 6, 6>::Zero();
    ext.diagonal() << Vector3d::Constant(kSigmaExtrinsicTrans * kSigmaExtrinsicTrans),
        Vector3d::Constant(kSigmaExtrinsicRot * kSigmaExtrinsicRot);

    vio::EqvioParams params;
    params.imu = config.noise;
    params.gravity_world = config.gravity_world;
    params.bearing_sigma = config.pixel_sigma / config.focal_px;
    vio::EqvioEstimator est(init, vio::eqvio::InitialCovariance(init, nav, bias, ext), params);
    vio::EskfEstimator dead_reckoning(vio::EskfState{init.p, init.v, init.R},
                                      Eigen::Matrix<double, 15, 15>::Zero(), config.noise,
                                      config.gravity_world);

    const double dt = 1.0 / config.imu_hz;
    std::size_t next_frame = 0, next_report = 0;
    for (std::size_t k = 0; k < sim.imu.size(); ++k) {
      if (next_frame < sim.frames.size() && sim.frames[next_frame].imu_index == k) {
        est.Update(sim.frames[next_frame].bearings);
        ++next_frame;
        tracked_sum += static_cast<double>(est.landmark_ids().size());
        ++tracked_count;
        if (next_report < report_times.size() && sim.imu[k].t >= report_times[next_report] - 1e-9) {
          const vio::EqvioState truth = vio::TruthState(sim, k);
          const vio::EqvioState s = est.state();
          const Eigen::VectorXd eps = est.ErrorCoordinates(truth);
          const Eigen::MatrixXd& P = est.covariance();
          Accumulator& a = acc[next_report];
          a.t = sim.imu[k].t;
          a.pos_sq += (s.p - truth.p).squaredNorm();
          a.att_sq += (s.R.inverse() * truth.R).log().squaredNorm();
          a.nees_full += Nees(eps, P) / static_cast<double>(eps.size());
          a.nees_pose += Nees(eps.head<6>(), P.topLeftCorner<6, 6>()) / 6.0;
          a.nees_att += Nees(eps.head<3>(), P.topLeftCorner<3, 3>()) / 3.0;
          ++a.count;
          ++next_report;
        }
      }
      if (k + 1 < sim.imu.size()) {
        est.Predict(sim.imu[k].gyro_meas, sim.imu[k].accel_meas, dt);
        dead_reckoning.Predict(sim.imu[k].gyro_meas, sim.imu[k].accel_meas, dt);
      }
    }

    const vio::EqvioState s = est.state();
    const vio::ImuTrajectorySample& last = sim.imu.back();
    final_gyro_bias_sq += (s.bias_gyro - last.gyro_bias_true).squaredNorm();
    final_accel_bias_sq += (s.bias_accel - last.accel_bias_true).squaredNorm();
    const vio::Vector6d ext_err = (s.T_body_camera.inverse() * sim.T_body_camera).log();
    final_ext_trans_sq += ext_err.head<3>().squaredNorm();
    final_ext_rot_sq += ext_err.tail<3>().squaredNorm();
    dr_final_sq += (dead_reckoning.state().p - last.p_true).squaredNorm();
  }

  const double kRadToDeg = 180.0 / 3.14159265358979323846;
  const double runs = opt.runs;
  std::cout << std::fixed;
  if (opt.markdown) {
    std::cout << "### EqVIO simulation (" << opt.runs << " Monte Carlo runs, " << opt.duration_s
              << " s)\n\n"
              << "| t [s] | pos RMSE [m] | att RMSE [deg] | ANEES full | ANEES pose | ANEES att |\n"
              << "|---|---|---|---|---|---|\n";
  } else {
    std::cout << "EqVIO simulation: " << opt.runs << " Monte Carlo runs, " << opt.duration_s
              << " s, mean tracked landmarks " << std::setprecision(1)
              << tracked_sum / tracked_count << "\n"
              << "ANEES = average NEES / dim (1 = consistent)\n\n"
              << "   t[s]  pos RMSE[m]  att RMSE[deg]  ANEES full  ANEES pose  ANEES att\n";
  }
  for (const Accumulator& a : acc) {
    if (a.count == 0) continue;
    const double n = a.count;
    if (opt.markdown) {
      std::cout << std::setprecision(1) << "| " << a.t << std::setprecision(4) << " | "
                << std::sqrt(a.pos_sq / n) << " | " << std::sqrt(a.att_sq / n) * kRadToDeg << " | "
                << std::setprecision(2) << a.nees_full / n << " | " << a.nees_pose / n << " | "
                << a.nees_att / n << " |\n";
    } else {
      std::cout << std::setprecision(1) << std::setw(7) << a.t << std::setprecision(4)
                << std::setw(13) << std::sqrt(a.pos_sq / n) << std::setw(15)
                << std::sqrt(a.att_sq / n) * kRadToDeg << std::setprecision(2) << std::setw(12)
                << a.nees_full / n << std::setw(12) << a.nees_pose / n << std::setw(11)
                << a.nees_att / n << "\n";
    }
  }
  std::cout << std::setprecision(4) << "\n" << "Final RMSE over runs:"
            << " gyro bias " << std::sqrt(final_gyro_bias_sq / runs) << " rad/s (prior "
            << kSigmaGyroBias * std::sqrt(3.0) << "),"
            << " accel bias " << std::sqrt(final_accel_bias_sq / runs) << " m/s^2 (prior "
            << kSigmaAccelBias * std::sqrt(3.0) << "),"
            << " extrinsic rot " << std::sqrt(final_ext_rot_sq / runs) * kRadToDeg << " deg (prior "
            << kSigmaExtrinsicRot * std::sqrt(3.0) * kRadToDeg << "),"
            << " extrinsic trans " << std::sqrt(final_ext_trans_sq / runs) << " m (prior "
            << kSigmaExtrinsicTrans * std::sqrt(3.0) << ")\n"
            << "IMU-only dead reckoning final position RMSE: " << std::sqrt(dr_final_sq / runs)
            << " m\n";
  return 0;
}

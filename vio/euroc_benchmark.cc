// Usage: euroc_benchmark <path/to/mav0> [--format=text|markdown]
//                        [--estimator=eskf|eqvio]
//
// Runs one VIO pipeline over one EuRoC sequence and reports its Absolute
// Trajectory Error (ATE) against ground truth:
//   eskf  (default): loosely-coupled ESKF + hand-rolled stereo VO.
//   eqvio: monocular EqVIO (vio/eqvio.h) on cam0 + a patch-tracking front end.
//          Tuning overrides: --features=N --bearing-sigma-px=S --chi2=T
//          --imu-noise-scale=K (multiplies all four EqvioPipelineOptions
//          IMU densities).
//
// --alignment=umeyama (default) | initial: how the estimate is aligned to
// ground truth before computing ATE; see vio::AteAlignment. Unlike
// vio_demo, this does not fall back to a default dataset location or
// gracefully no-op when the sequence is missing -- it's meant to be driven
// explicitly by a script (e.g. CI), so a missing/malformed sequence is a
// hard failure (nonzero exit), not a friendly notice.
//
// --format=text (default): human-readable report on stdout.
// --format=markdown: a ready-to-post Markdown table on stdout, suitable for
// `euroc_benchmark <mav0> --format=markdown > ate_report.md` followed by
// posting ate_report.md as a PR comment.
#include <iomanip>
#include <iostream>
#include <sstream>

#include "eqvio_pipeline.h"
#include "euroc_loader.h"
#include "imu_trajectory_simulator.h"
#include "trajectory_evaluation.h"

namespace {

void PrintUsage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " <path/to/mav0> [--format=text|markdown] [--estimator=eskf|eqvio]\n";
}

std::string FormatMeters(double value) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4) << value;
  return oss.str();
}

void PrintText(const std::string& mav0_dir, const vio::EurocSequence& seq,
              const vio::PipelineStats& stats, const vio::AteResult& ate) {
  std::cout << "EuRoC benchmark: " << mav0_dir << "\n"
            << "  " << seq.imu_samples.size() << " IMU samples, " << seq.cam0_frames.size()
            << " cam0 frames, " << seq.ground_truth.size() << " ground-truth samples\n"
            << "  Gravity sanity check (expect near (0,0,9.81)): [" << stats.gravity_check.x()
            << ", " << stats.gravity_check.y() << ", " << stats.gravity_check.z() << "]\n";
  if (stats.unpaired_cam0_frames > 0) {
    std::cout << "  Warning: " << stats.unpaired_cam0_frames
              << " cam0 frames had no exact cam1 timestamp match\n";
  }
  std::cout << "\n"
            << "Camera frames processed: " << stats.camera_frames_processed << "\n"
            << "Average tracked points/frame: " << stats.avg_tracked_points << "\n"
            << "Average RANSAC inliers/frame: " << stats.avg_inliers << "\n"
            << "ATE samples: " << ate.num_samples << "\n"
            << "ATE RMSE:   " << FormatMeters(ate.rmse_m) << " m\n"
            << "ATE mean:   " << FormatMeters(ate.mean_m) << " m\n"
            << "ATE median: " << FormatMeters(ate.median_m) << " m\n"
            << "ATE max:    " << FormatMeters(ate.max_m) << " m\n";
}

void PrintMarkdown(const std::string& mav0_dir, const vio::EurocSequence& seq,
                   const vio::PipelineStats& stats, const vio::AteResult& ate) {
  std::cout << "### EuRoC VIO benchmark: `" << mav0_dir << "`\n\n"
            << "| Metric | Value |\n"
            << "|---|---|\n"
            << "| ATE RMSE | " << FormatMeters(ate.rmse_m) << " m |\n"
            << "| ATE mean | " << FormatMeters(ate.mean_m) << " m |\n"
            << "| ATE median | " << FormatMeters(ate.median_m) << " m |\n"
            << "| ATE max | " << FormatMeters(ate.max_m) << " m |\n"
            << "| Trajectory samples | " << ate.num_samples << " |\n"
            << "| Camera frames processed | " << stats.camera_frames_processed << " |\n"
            << "| Avg. tracked points/frame | " << stats.avg_tracked_points << " |\n"
            << "| Avg. RANSAC inliers/frame | " << stats.avg_inliers << " |\n"
            << "| IMU samples | " << seq.imu_samples.size() << " |\n"
            << "| Ground-truth samples | " << seq.ground_truth.size() << " |\n";
  if (stats.unpaired_cam0_frames > 0) {
    std::cout << "\n:warning: " << stats.unpaired_cam0_frames
              << " cam0 frames had no exact cam1 timestamp match.\n";
  }
}

void PrintEqvioText(const std::string& mav0_dir, const vio::EurocSequence& seq,
                    const vio::EqvioPipelineStats& stats, const vio::AteResult& ate) {
  std::cout << "EuRoC benchmark (EqVIO, monocular cam0): " << mav0_dir << "\n"
            << "  " << seq.imu_samples.size() << " IMU samples, " << seq.cam0_frames.size()
            << " cam0 frames, " << seq.ground_truth.size() << " ground-truth samples\n\n"
            << "Camera frames processed: " << stats.camera_frames_processed << "\n"
            << "Average tracked features/frame: " << stats.avg_tracked_features << "\n"
            << "Average landmarks in update/frame: " << stats.avg_landmarks_used << "\n"
            << "Outliers rejected: " << stats.outliers_rejected << "\n"
            << "ATE samples: " << ate.num_samples << "\n"
            << "ATE RMSE:   " << FormatMeters(ate.rmse_m) << " m\n"
            << "ATE mean:   " << FormatMeters(ate.mean_m) << " m\n"
            << "ATE median: " << FormatMeters(ate.median_m) << " m\n"
            << "ATE max:    " << FormatMeters(ate.max_m) << " m\n";
}

void PrintEqvioMarkdown(const std::string& mav0_dir, const vio::EqvioPipelineStats& stats,
                        const vio::AteResult& ate) {
  std::cout << "### EuRoC EqVIO benchmark (monocular cam0): `" << mav0_dir << "`\n\n"
            << "| Metric | Value |\n"
            << "|---|---|\n"
            << "| ATE RMSE | " << FormatMeters(ate.rmse_m) << " m |\n"
            << "| ATE mean | " << FormatMeters(ate.mean_m) << " m |\n"
            << "| ATE median | " << FormatMeters(ate.median_m) << " m |\n"
            << "| ATE max | " << FormatMeters(ate.max_m) << " m |\n"
            << "| Trajectory samples | " << ate.num_samples << " |\n"
            << "| Camera frames processed | " << stats.camera_frames_processed << " |\n"
            << "| Avg. tracked features/frame | " << stats.avg_tracked_features << " |\n"
            << "| Avg. landmarks in update/frame | " << stats.avg_landmarks_used << " |\n"
            << "| Outliers rejected | " << stats.outliers_rejected << " |\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string mav0_dir;
  std::string format = "text";
  std::string estimator = "eskf";
  vio::EqvioPipelineOptions eqvio_options;
  double imu_noise_scale = 1.0;
  vio::AteAlignment alignment = vio::AteAlignment::kUmeyama;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("--format=", 0) == 0) {
      format = arg.substr(std::string("--format=").size());
    } else if (arg.rfind("--estimator=", 0) == 0) {
      estimator = arg.substr(std::string("--estimator=").size());
    } else if (arg == "--alignment=initial") {
      alignment = vio::AteAlignment::kInitialState;
    } else if (arg == "--alignment=umeyama") {
      alignment = vio::AteAlignment::kUmeyama;
    } else if (arg.rfind("--features=", 0) == 0) {
      eqvio_options.tracker.max_features = std::stoi(arg.substr(11));
    } else if (arg.rfind("--bearing-sigma-px=", 0) == 0) {
      eqvio_options.bearing_sigma_px = std::stod(arg.substr(19));
    } else if (arg.rfind("--chi2=", 0) == 0) {
      eqvio_options.outlier_chi2_threshold = std::stod(arg.substr(7));
    } else if (arg.rfind("--imu-noise-scale=", 0) == 0) {
      imu_noise_scale = std::stod(arg.substr(18));
    } else if (mav0_dir.empty()) {
      mav0_dir = arg;
    } else {
      PrintUsage(argv[0]);
      return 1;
    }
  }

  if (mav0_dir.empty() || (format != "text" && format != "markdown") ||
      (estimator != "eskf" && estimator != "eqvio")) {
    PrintUsage(argv[0]);
    return 1;
  }

  const std::string resolved = vio::FindEurocSequence(mav0_dir, {});
  if (resolved.empty()) {
    std::cerr << "EuRoC sequence not found or unreadable at: " << mav0_dir << "\n";
    return 1;
  }

  vio::EurocSequence seq;
  try {
    seq = vio::LoadEurocSequence(resolved);
  } catch (const std::exception& e) {
    std::cerr << "Failed to load EuRoC sequence: " << e.what() << "\n";
    return 1;
  }

  if (estimator == "eqvio") {
    vio::EqvioPipelineStats stats;
    std::vector<vio::TrajectorySample> trajectory;
    try {
      eqvio_options.imu_noise = vio::ScaleNoise(eqvio_options.imu_noise, imu_noise_scale);
      trajectory = vio::RunEqvioPipeline(seq, resolved, eqvio_options, &stats);
    } catch (const std::exception& e) {
      std::cerr << "Pipeline failed: " << e.what() << "\n";
      return 1;
    }
    if (trajectory.size() < 3) {
      std::cerr << "Pipeline produced too few trajectory samples (" << trajectory.size()
                << ") to compute ATE (need >= 3).\n";
      return 1;
    }
    const vio::AteResult ate = vio::ComputeAte(trajectory, alignment);
    if (format == "markdown") {
      PrintEqvioMarkdown(resolved, stats, ate);
    } else {
      PrintEqvioText(resolved, seq, stats, ate);
    }
    return 0;
  }

  vio::PipelineOptions options;
  vio::PipelineStats stats;
  std::vector<vio::TrajectorySample> trajectory;
  try {
    trajectory = vio::RunPipeline(seq, resolved, options, &stats);
  } catch (const std::exception& e) {
    std::cerr << "Pipeline failed: " << e.what() << "\n";
    return 1;
  }

  if (trajectory.size() < 3) {
    std::cerr << "Pipeline produced too few trajectory samples (" << trajectory.size()
              << ") to compute ATE (need >= 3).\n";
    return 1;
  }

  const vio::AteResult ate = vio::ComputeAte(trajectory, alignment);

  if (format == "markdown") {
    PrintMarkdown(resolved, seq, stats, ate);
  } else {
    PrintText(resolved, seq, stats, ate);
  }

  return 0;
}

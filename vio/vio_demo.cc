// Usage: bazel run //vio:vio_demo -- [path/to/mav0]
//
// Runs the full loosely-coupled ESKF + hand-rolled stereo VO pipeline over
// one EuRoC sequence and reports its Absolute Trajectory Error (ATE)
// against the dataset's own ground truth. If mav0 is omitted, tries a few
// candidate paths; if none exist, prints the commands to prepare the data
// from the already-downloaded data/EuRoC/vicon_room1.zip and exits 0 (no
// hard crash).
#include <chrono>
#include <iostream>

#include "euroc_loader.h"
#include "trajectory_evaluation.h"

namespace {

void PrintPrepInstructions() {
  std::cout << "EuRoC sequence not found. Prepare it once with:\n\n"
            << "  mkdir -p data/EuRoC/vicon_room1/V1_01_easy\n"
            << "  unzip -j data/EuRoC/vicon_room1.zip "
               "'vicon_room1/V1_01_easy/V1_01_easy.zip' -d "
               "data/EuRoC/vicon_room1/V1_01_easy\n"
            << "  unzip data/EuRoC/vicon_room1/V1_01_easy/V1_01_easy.zip -d "
               "data/EuRoC/vicon_room1/V1_01_easy\n"
            << "  rm data/EuRoC/vicon_room1/V1_01_easy/V1_01_easy.zip\n\n"
            << "then rerun: bazel run //vio:vio_demo\n";
}

}  // namespace

int main(int argc, char** argv) {
  const std::string explicit_path = argc > 1 ? argv[1] : "";
  const std::string mav0_dir =
      vio::FindEurocSequence(explicit_path, {"data/EuRoC/vicon_room1/V1_01_easy/mav0",
                                             "../data/EuRoC/vicon_room1/V1_01_easy/mav0"});
  if (mav0_dir.empty()) {
    PrintPrepInstructions();
    return 0;
  }

  std::cout << "Loading EuRoC sequence from " << mav0_dir << " ...\n" << std::flush;
  const vio::EurocSequence seq = vio::LoadEurocSequence(mav0_dir);
  std::cout << "  " << seq.imu_samples.size() << " IMU samples, " << seq.cam0_frames.size()
            << " cam0 frames, " << seq.ground_truth.size() << " ground-truth samples\n"
            << std::flush;

  vio::PipelineOptions options;
  vio::PipelineStats stats;

  const auto wall_start = std::chrono::steady_clock::now();
  std::vector<vio::TrajectorySample> trajectory;
  try {
    trajectory = vio::RunPipeline(seq, mav0_dir, options, &stats, [&](long done, long total) {
      const double elapsed_s =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
      std::cout << "  frame " << done << "/" << total << " (" << elapsed_s << "s elapsed, "
                << (elapsed_s / done) << "s/frame)\n"
                << std::flush;
    });
  } catch (const std::exception& e) {
    std::cerr << "Pipeline failed: " << e.what() << "\n";
    return 1;
  }

  if (trajectory.empty()) {
    std::cerr << "Pipeline produced no trajectory samples (missing IMU/camera/ground-truth "
                 "data?).\n";
    return 1;
  }

  const vio::AteResult ate = vio::ComputeAte(trajectory);

  std::cout << "\nGravity sanity check (first ~1s, expect near (0,0,9.81)): ["
            << stats.gravity_check.x() << ", " << stats.gravity_check.y() << ", "
            << stats.gravity_check.z() << "]\n";
  if (stats.unpaired_cam0_frames > 0) {
    std::cout << "Warning: " << stats.unpaired_cam0_frames
              << " cam0 frames had no exact cam1 timestamp match\n";
  }

  std::cout << "\n=== Results ===\n"
            << "Camera frames processed: " << stats.camera_frames_processed << "\n"
            << "Average tracked points/frame: " << stats.avg_tracked_points << "\n"
            << "Average RANSAC inliers/frame: " << stats.avg_inliers << "\n"
            << "ATE RMSE:   " << ate.rmse_m << " m\n"
            << "ATE mean:   " << ate.mean_m << " m\n"
            << "ATE median: " << ate.median_m << " m\n"
            << "ATE max:    " << ate.max_m << " m\n";

  return 0;
}

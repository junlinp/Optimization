#ifndef VIO_TRAJECTORY_EVALUATION_H_
#define VIO_TRAJECTORY_EVALUATION_H_
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "euroc_types.h"
#include "harris_corners.h"
#include "patch_matcher.h"
#include "rigid_registration.h"

namespace vio {

struct TrajectorySample {
  int64_t timestamp_ns = 0;
  Eigen::Vector3d p_est = Eigen::Vector3d::Zero();
  Eigen::Vector3d p_gt = Eigen::Vector3d::Zero();
};

struct PipelineOptions {
  Eigen::Vector3d gravity_world = Eigen::Vector3d(0, 0, -9.81);
  // The datasheet IMU noise densities alone make the filter's self-reported
  // uncertainty settle far below the VO measurement noise within ~100
  // frames, so under 10% of every VO correction gets trusted and the
  // estimate effectively free-runs on (drifting) IMU dead reckoning.
  // Datasheet sensor noise is well known to be overly optimistic once
  // vibration, bias instability beyond a simple random walk, and
  // linearization error are accounted for; inflating it with a tuning
  // safety factor is standard VIO practice, not a hack. 5000x was found by
  // a measured sweep (1x/50x/250x/1000x/5000x) on V1_01_easy: it's the
  // point where the filter's covariance reaches parity with the assumed
  // measurement noise (see RelativePoseMeasurement's sigma_* defaults) --
  // final position error dropped from 46.5m to 3.5m there, with rapidly
  // diminishing returns beyond it (further inflation just makes the
  // estimate echo the VO front end's own ~15-20% path-length bias).
  double process_noise_inflation = 5000.0;
  HarrisOptions harris_options;
  PatchMatchOptions match_options;
  RansacOptions ransac_options;
};

struct PipelineStats {
  long camera_frames_processed = 0;
  int unpaired_cam0_frames = 0;
  double avg_tracked_points = 0;
  double avg_inliers = 0;
  // Average R_gt * accel_meas over the first ~1s of (near-static) data.
  // Should read close to (0,0,+9.81); if it instead reads close to
  // (0,0,-9.81), PipelineOptions::gravity_world's sign should be flipped.
  Eigen::Vector3d gravity_check = Eigen::Vector3d::Zero();
};

// Runs the loosely-coupled ESKF + hand-rolled stereo VO pipeline over the
// whole sequence: initializes at the first camera-frame timestamp (seeded
// from the nearest ground-truth sample), then IMU-predicts and VO-updates
// through every event in timestamp order. Returns one TrajectorySample per
// successfully processed camera frame (estimated position paired with the
// nearest ground-truth position at that timestamp).
//
// progress_callback, if set, is invoked with (frames_processed, total_frames)
// periodically during the run.
std::vector<TrajectorySample> RunPipeline(
    const EurocSequence& seq, const std::string& mav0_dir, const PipelineOptions& options,
    PipelineStats* stats,
    const std::function<void(long, long)>& progress_callback = nullptr);

struct AteResult {
  double rmse_m = 0;
  double mean_m = 0;
  double median_m = 0;
  double max_m = 0;
  int num_samples = 0;
};

// Absolute Trajectory Error (Sturm et al., "A Benchmark for the Evaluation
// of RGB-D SLAM Systems", 2012): rigidly aligns the estimated trajectory
// onto ground truth (translation + rotation, no scale -- appropriate for a
// metrically-scaled stereo+IMU system, unlike monocular-only SLAM) via
// closed-form Umeyama registration over the whole trajectory, then reports
// RMSE/mean/median/max of the aligned position error. Requires
// samples.size() >= 3 (Umeyama's minimum).
AteResult ComputeAte(const std::vector<TrajectorySample>& samples);

// Candidate mav0/ directory paths to probe, in order; the first containing
// a readable cam0/data.csv is returned. Returns "" if none match. Tries
// explicit_path first if non-empty.
std::string FindEurocSequence(const std::string& explicit_path,
                              const std::vector<std::string>& default_candidates);

}  // namespace vio
#endif  // VIO_TRAJECTORY_EVALUATION_H_

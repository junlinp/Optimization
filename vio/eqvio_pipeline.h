#ifndef VIO_EQVIO_PIPELINE_H_
#define VIO_EQVIO_PIPELINE_H_
#include <functional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "eskf_estimator.h"
#include "euroc_types.h"
#include "mono_feature_tracker.h"
#include "trajectory_evaluation.h"

namespace vio {

struct EqvioPipelineOptions {
  Eigen::Vector3d gravity_world = Eigen::Vector3d(0, 0, -9.81);
  // The paper's EuRoC tuning (van Goor & Mahony, Table 2) rather than the
  // datasheet densities in imu0/sensor.yaml, which are optimistic (see
  // PipelineOptions::process_noise_inflation): 1.4x the datasheet gyro
  // noise, 6x accel noise, 7x gyro bias walk, 1.5x accel bias walk.
  ImuNoiseParams imu_noise = [] {
    ImuNoiseParams n;
    n.gyro_noise_density = 2.43e-4;
    n.accel_noise_density = 1.24e-2;
    n.gyro_random_walk = 1.34e-4;
    n.accel_random_walk = 4.46e-3;
    return n;
  }();
  // Paper Table 2. A sweep on V1_01/02/03 found no better value for this
  // front end (mean ATE RMSE 0.142 m at 1.9 px, 0.144 m at 2.5 px, 0.158 m
  // at 3.0 px; neighbouring settings differ by about the run-to-run spread).
  double bearing_sigma_px = 1.9;
  double initial_landmark_depth_m = 3.0;
  // 2-dof 99%; <= 0 disables. Required with this front end: without it the
  // occasional mis-track drives the filter to divergence on V1_01.
  double outlier_chi2_threshold = 9.21;
  MonoTrackerOptions tracker = [] {
    MonoTrackerOptions o;
    o.max_features = 40;  // paper Table 5
    return o;
  }();
};

struct EqvioPipelineStats {
  long camera_frames_processed = 0;
  double avg_tracked_features = 0;  // front-end tracks per frame
  double avg_landmarks_used = 0;    // landmarks in the EqF update per frame
  long outliers_rejected = 0;
};

// Monocular EqVIO on cam0 + imu0. Same protocol as RunPipeline (the ESKF +
// stereo VO baseline) so their ATEs are comparable: the filter starts at the
// first cam0 frame, seeded from the nearest ground-truth pose, velocity and
// biases, with the cam0 extrinsic from sensor.yaml; one TrajectorySample is
// returned per processed frame.
std::vector<TrajectorySample> RunEqvioPipeline(
    const EurocSequence& seq, const std::string& mav0_dir, const EqvioPipelineOptions& options,
    EqvioPipelineStats* stats,
    const std::function<void(long, long)>& progress_callback = nullptr);

}  // namespace vio
#endif  // VIO_EQVIO_PIPELINE_H_

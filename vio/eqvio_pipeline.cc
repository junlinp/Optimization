#include "eqvio_pipeline.h"

#include "eqvio.h"
#include "image_io.h"

namespace vio {

std::vector<TrajectorySample> RunEqvioPipeline(
    const EurocSequence& seq, const std::string& mav0_dir, const EqvioPipelineOptions& options,
    EqvioPipelineStats* stats, const std::function<void(long, long)>& progress_callback) {
  *stats = EqvioPipelineStats();
  std::vector<TrajectorySample> trajectory;
  if (seq.imu_samples.empty() || seq.cam0_frames.empty() || seq.ground_truth.empty()) {
    return trajectory;
  }

  const int64_t t_start = seq.cam0_frames.front().timestamp_ns;
  const GroundTruthSample& gt0 = NearestGroundTruth(seq, t_start);
  EqvioState x0;
  x0.R = gt0.R_world_body;
  x0.p = gt0.p_world;
  x0.v = gt0.v_world;
  x0.bias_gyro = gt0.bias_gyro;
  x0.bias_accel = gt0.bias_accel;
  x0.T_body_camera = seq.cam0.T_BS;

  Eigen::Matrix<double, 9, 9> nav = Eigen::Matrix<double, 9, 9>::Zero();
  nav.diagonal() << Eigen::Vector3d::Constant(1e-6), Eigen::Vector3d::Constant(1e-6),
      Eigen::Vector3d::Constant(1e-4);
  Eigen::Matrix<double, 6, 6> bias = Eigen::Matrix<double, 6, 6>::Zero();
  bias.diagonal() << Eigen::Vector3d::Constant(1e-6), Eigen::Vector3d::Constant(1e-4);
  const Eigen::Matrix<double, 6, 6> extrinsic = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;

  EqvioParams params;
  params.gravity_world = options.gravity_world;
  params.imu = options.imu_noise;
  params.bearing_sigma = options.bearing_sigma_px / seq.cam0.intrinsics.fu;
  params.initial_landmark_depth = options.initial_landmark_depth_m;
  params.outlier_chi2_threshold = options.outlier_chi2_threshold;
  EqvioEstimator estimator(x0, eqvio::InitialCovariance(x0, nav, bias, extrinsic), params);

  MonoFeatureTracker tracker(seq.cam0, options.tracker);

  // IMU samples strictly before the first frame are skipped: the filter
  // starts at t_start. Each sample at t_k is applied over (t_{k-1}, t_k], as
  // in RunPipeline, and IMU samples are processed before a frame with the
  // same timestamp.
  std::size_t imu_index = 0;
  while (imu_index < seq.imu_samples.size() &&
         seq.imu_samples[imu_index].timestamp_ns <= t_start) {
    ++imu_index;
  }
  int64_t last_imu_ts = t_start;

  Sophus::SO3d R_world_prevcam = x0.T_world_camera().so3();
  long total_tracked = 0, total_used = 0;
  for (const CameraFrameEntry& frame : seq.cam0_frames) {
    while (imu_index < seq.imu_samples.size() &&
           seq.imu_samples[imu_index].timestamp_ns <= frame.timestamp_ns) {
      const ImuSample& s = seq.imu_samples[imu_index];
      const double dt = static_cast<double>(s.timestamp_ns - last_imu_ts) * 1e-9;
      if (dt > 0) estimator.Predict(s.gyro, s.accel, dt);
      last_imu_ts = s.timestamp_ns;
      ++imu_index;
    }

    // The filter's propagated camera rotation since the last frame is the
    // tracker's motion prior.
    const Sophus::SO3d R_world_cam = estimator.state().T_world_camera().so3();
    const GrayImage image = LoadGrayscalePng(mav0_dir + "/cam0/data/" + frame.filename);
    const std::vector<TrackedFeature> features =
        tracker.Track(image, R_world_prevcam.inverse() * R_world_cam);

    std::vector<BearingMeasurement> measurements;
    measurements.reserve(features.size());
    for (const TrackedFeature& f : features) measurements.push_back({f.id, f.bearing});
    const EqvioEstimator::UpdateResult result = estimator.Update(measurements);

    total_tracked += static_cast<long>(features.size());
    total_used += result.num_used;
    stats->outliers_rejected += result.num_outliers;
    ++stats->camera_frames_processed;
    if (progress_callback && stats->camera_frames_processed % 100 == 0) {
      progress_callback(stats->camera_frames_processed, static_cast<long>(seq.cam0_frames.size()));
    }

    const EqvioState state = estimator.state();
    R_world_prevcam = state.T_world_camera().so3();
    trajectory.push_back(
        {frame.timestamp_ns, state.p, NearestGroundTruth(seq, frame.timestamp_ns).p_world});
  }

  if (stats->camera_frames_processed > 0) {
    stats->avg_tracked_features =
        static_cast<double>(total_tracked) / stats->camera_frames_processed;
    stats->avg_landmarks_used = static_cast<double>(total_used) / stats->camera_frames_processed;
  }
  return trajectory;
}

}  // namespace vio

#include "trajectory_evaluation.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <unordered_map>

#include "eskf_estimator.h"
#include "euroc_loader.h"
#include "image_io.h"
#include "stereo_rectifier.h"
#include "stereo_vo_frontend.h"

namespace vio {
namespace {

// seq.ground_truth is sorted by timestamp_ns; binary search for the entry
// closest to `timestamp_ns`.
const GroundTruthSample& NearestGroundTruth(const EurocSequence& seq, int64_t timestamp_ns) {
  const auto& gt = seq.ground_truth;
  auto it = std::lower_bound(gt.begin(), gt.end(), timestamp_ns,
                             [](const GroundTruthSample& sample, int64_t t) {
                               return sample.timestamp_ns < t;
                             });
  if (it == gt.begin()) return gt.front();
  if (it == gt.end()) return gt.back();
  const auto prev = it - 1;
  return (timestamp_ns - prev->timestamp_ns <= it->timestamp_ns - timestamp_ns) ? *prev : *it;
}

struct Event {
  int64_t timestamp_ns = 0;
  bool is_camera = false;
  size_t index = 0;
};

struct FramePair {
  int64_t timestamp_ns;
  std::string cam0_filename, cam1_filename;
};

}  // namespace

std::vector<TrajectorySample> RunPipeline(const EurocSequence& seq, const std::string& mav0_dir,
                                          const PipelineOptions& options, PipelineStats* stats,
                                          const std::function<void(long, long)>& progress_callback) {
  *stats = PipelineStats();
  std::vector<TrajectorySample> trajectory;

  std::unordered_map<int64_t, std::string> cam1_by_ts;
  for (const CameraFrameEntry& f : seq.cam1_frames) cam1_by_ts[f.timestamp_ns] = f.filename;

  std::vector<FramePair> frame_pairs;
  for (const CameraFrameEntry& f0 : seq.cam0_frames) {
    auto it = cam1_by_ts.find(f0.timestamp_ns);
    if (it == cam1_by_ts.end()) {
      ++stats->unpaired_cam0_frames;
      continue;
    }
    frame_pairs.push_back({f0.timestamp_ns, f0.filename, it->second});
  }

  if (seq.imu_samples.empty() || frame_pairs.empty() || seq.ground_truth.empty()) {
    return trajectory;
  }

  std::vector<Event> events;
  events.reserve(seq.imu_samples.size() + frame_pairs.size());
  for (size_t i = 0; i < seq.imu_samples.size(); ++i) {
    events.push_back({seq.imu_samples[i].timestamp_ns, false, i});
  }
  for (size_t i = 0; i < frame_pairs.size(); ++i) {
    events.push_back({frame_pairs[i].timestamp_ns, true, i});
  }
  std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) {
    if (a.timestamp_ns != b.timestamp_ns) return a.timestamp_ns < b.timestamp_ns;
    return !a.is_camera && b.is_camera;  // IMU first on exact ties
  });

  // Gravity sign sanity check: average R_gt * accel_meas over the first ~1s
  // of data (near-static).
  {
    const int64_t t0 = seq.imu_samples.front().timestamp_ns;
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    int count = 0;
    for (const auto& sample : seq.imu_samples) {
      if (sample.timestamp_ns - t0 > 1000000000LL) break;
      const auto& gt = NearestGroundTruth(seq, sample.timestamp_ns);
      sum += gt.R_world_body.matrix() * sample.accel;
      ++count;
    }
    stats->gravity_check = count > 0 ? (sum / count).eval() : Eigen::Vector3d::Zero();
  }

  const GroundTruthSample& gt0 = NearestGroundTruth(seq, frame_pairs.front().timestamp_ns);
  EskfState x0;
  x0.p = gt0.p_world;
  x0.v = gt0.v_world;
  x0.R = gt0.R_world_body;
  x0.bias_gyro = gt0.bias_gyro;
  x0.bias_accel = gt0.bias_accel;

  Eigen::Matrix<double, 15, 15> P0 = Eigen::Matrix<double, 15, 15>::Zero();
  P0.block<9, 9>(0, 0) = Eigen::Matrix<double, 9, 9>::Identity() * 1e-4;
  P0.block<6, 6>(9, 9) = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;

  ImuNoiseParams noise;
  noise.gyro_noise_density = seq.imu0.gyro_noise_density * options.process_noise_inflation;
  noise.gyro_random_walk = seq.imu0.gyro_random_walk * options.process_noise_inflation;
  noise.accel_noise_density = seq.imu0.accel_noise_density * options.process_noise_inflation;
  noise.accel_random_walk = seq.imu0.accel_random_walk * options.process_noise_inflation;

  EskfEstimator estimator(x0, P0, noise, options.gravity_world);

  const StereoRectification rectification =
      ComputeStereoRectification(seq.cam0, seq.cam1, seq.cam0.height, seq.cam0.width);
  StereoVoFrontend frontend(rectification, options.harris_options, options.match_options,
                            options.ransac_options);

  int64_t last_imu_ts = -1;
  bool first_camera_frame = true;
  long total_tracked = 0, total_inliers = 0;

  for (const Event& event : events) {
    if (!event.is_camera) {
      const auto& sample = seq.imu_samples[event.index];
      if (last_imu_ts >= 0) {
        const double dt = static_cast<double>(sample.timestamp_ns - last_imu_ts) * 1e-9;
        estimator.Predict(sample.gyro, sample.accel, dt);
      }
      last_imu_ts = sample.timestamp_ns;
      continue;
    }

    const FramePair& pair = frame_pairs[event.index];
    const GrayImage cam0_img = LoadGrayscalePng(mav0_dir + "/cam0/data/" + pair.cam0_filename);
    const GrayImage cam1_img = LoadGrayscalePng(mav0_dir + "/cam1/data/" + pair.cam1_filename);

    const StereoVoFrontend::FrameResult vo_result = frontend.ProcessFrame(cam0_img, cam1_img);
    total_tracked += vo_result.num_tracked;
    total_inliers += vo_result.num_inliers;
    ++stats->camera_frames_processed;

    if (progress_callback && stats->camera_frames_processed % 100 == 0) {
      progress_callback(stats->camera_frames_processed, static_cast<long>(frame_pairs.size()));
    }

    if (first_camera_frame) {
      estimator.SetRelativePoseAnchor();
      first_camera_frame = false;
    } else {
      if (vo_result.has_relative_pose) {
        RelativePoseMeasurement measurement;
        measurement.T_prevbody_currbody = vo_result.T_prevbody_currbody;
        estimator.UpdateRelativePose(measurement);
      }
      estimator.SetRelativePoseAnchor();
    }

    const GroundTruthSample& gt = NearestGroundTruth(seq, pair.timestamp_ns);
    trajectory.push_back({pair.timestamp_ns, estimator.state().p, gt.p_world});
  }

  stats->avg_tracked_points = stats->camera_frames_processed > 0
                                 ? static_cast<double>(total_tracked) / stats->camera_frames_processed
                                 : 0.0;
  stats->avg_inliers = stats->camera_frames_processed > 0
                          ? static_cast<double>(total_inliers) / stats->camera_frames_processed
                          : 0.0;
  return trajectory;
}

AteResult ComputeAte(const std::vector<TrajectorySample>& samples) {
  AteResult result;
  result.num_samples = static_cast<int>(samples.size());
  if (samples.size() < 3) return result;

  std::vector<Eigen::Vector3d> est, gt;
  est.reserve(samples.size());
  gt.reserve(samples.size());
  for (const auto& s : samples) {
    est.push_back(s.p_est);
    gt.push_back(s.p_gt);
  }

  const RigidTransform alignment = UmeyamaAlignment(est, gt);

  std::vector<double> errors;
  errors.reserve(samples.size());
  double sum_sq = 0.0, sum = 0.0, max_error = 0.0;
  for (size_t i = 0; i < samples.size(); ++i) {
    const Eigen::Vector3d aligned = alignment.R * est[i] + alignment.t;
    const double error = (aligned - gt[i]).norm();
    errors.push_back(error);
    sum_sq += error * error;
    sum += error;
    max_error = std::max(max_error, error);
  }

  std::sort(errors.begin(), errors.end());
  const size_t mid = errors.size() / 2;
  const double median =
      (errors.size() % 2 == 0) ? 0.5 * (errors[mid - 1] + errors[mid]) : errors[mid];

  result.rmse_m = std::sqrt(sum_sq / samples.size());
  result.mean_m = sum / samples.size();
  result.median_m = median;
  result.max_m = max_error;
  return result;
}

std::string FindEurocSequence(const std::string& explicit_path,
                              const std::vector<std::string>& default_candidates) {
  std::vector<std::string> candidates;
  if (!explicit_path.empty()) candidates.push_back(explicit_path);
  candidates.insert(candidates.end(), default_candidates.begin(), default_candidates.end());

  for (const auto& path : candidates) {
    std::ifstream probe(path + "/cam0/data.csv");
    if (probe.good()) return path;
  }
  return "";
}

}  // namespace vio

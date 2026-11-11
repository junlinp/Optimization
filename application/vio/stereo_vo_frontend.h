#ifndef VIO_STEREO_VO_FRONTEND_H_
#define VIO_STEREO_VO_FRONTEND_H_
#include <vector>

#include "harris_corners.h"
#include "image_io.h"
#include "patch_matcher.h"
#include "rigid_registration.h"
#include "sophus/se3.hpp"
#include "stereo_rectifier.h"

namespace vio {

// KLT-style stereo visual-odometry front end. Each frame, previously
// tracked points are advanced via 2D temporal patch search, then
// independently re-stereo-matched and re-triangulated at the current frame;
// the resulting (previous-3D-point, current-3D-point) pairs feed
// RANSAC+Umeyama. Fresh Harris detection each frame only tops up points lost
// to tracking failure.
class StereoVoFrontend {
 public:
  struct FrameResult {
    bool has_relative_pose = false;
    Sophus::SE3d T_prevbody_currbody;
    int num_tracked = 0;
    int num_inliers = 0;
  };

  StereoVoFrontend(const StereoRectification& rectification, const HarrisOptions& harris_options,
                   const PatchMatchOptions& match_options, const RansacOptions& ransac_options);

  // Call once per stereo frame pair, strictly in increasing timestamp order.
  FrameResult ProcessFrame(const GrayImage& cam0_raw, const GrayImage& cam1_raw);

 private:
  struct TrackedPoint {
    double u_rect = 0, v_rect = 0;  // rectified-cam0 pixel, current frame
    Eigen::Vector3d p_cam0;         // 3D point in rectified-cam0 frame, current frame
  };

  std::vector<TrackedPoint> DetectAndTriangulateNew(const GrayImage& cam0_rect,
                                                    const GrayImage& cam1_rect,
                                                    const std::vector<TrackedPoint>& existing) const;

  StereoRectification rectification_;
  HarrisOptions harris_options_;
  PatchMatchOptions match_options_;
  RansacOptions ransac_options_;
  std::vector<TrackedPoint> tracked_points_;
  GrayImage prev_cam0_rectified_;
  bool have_prev_frame_ = false;
};

}  // namespace vio
#endif  // VIO_STEREO_VO_FRONTEND_H_

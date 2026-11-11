#include "stereo_vo_frontend.h"

#include <algorithm>
#include <set>
#include <utility>

#include "triangulation.h"

namespace vio {

StereoVoFrontend::StereoVoFrontend(const StereoRectification& rectification,
                                   const HarrisOptions& harris_options,
                                   const PatchMatchOptions& match_options,
                                   const RansacOptions& ransac_options)
    : rectification_(rectification),
      harris_options_(harris_options),
      match_options_(match_options),
      ransac_options_(ransac_options) {}

std::vector<StereoVoFrontend::TrackedPoint> StereoVoFrontend::DetectAndTriangulateNew(
    const GrayImage& cam0_rect, const GrayImage& cam1_rect,
    const std::vector<TrackedPoint>& existing) const {
  std::vector<TrackedPoint> added;
  const int budget = harris_options_.max_total_features - static_cast<int>(existing.size());
  if (budget <= 0) return added;

  const int cell_size = std::max(1, harris_options_.cell_size);
  auto CellOf = [cell_size](double x, double y) {
    return std::make_pair(static_cast<int>(y) / cell_size, static_cast<int>(x) / cell_size);
  };

  std::set<std::pair<int, int>> occupied;
  for (const TrackedPoint& p : existing) occupied.insert(CellOf(p.u_rect, p.v_rect));

  std::vector<Corner> corners = DetectHarrisCorners(cam0_rect, harris_options_);
  std::sort(corners.begin(), corners.end(),
           [](const Corner& a, const Corner& b) { return a.score > b.score; });

  for (const Corner& corner : corners) {
    if (static_cast<int>(added.size()) >= budget) break;

    const auto cell = CellOf(corner.x, corner.y);
    if (occupied.count(cell)) continue;

    double disparity = 0, stereo_score = 0;
    if (!MatchStereoPatch(cam0_rect, cam1_rect, corner.x, corner.y, match_options_, &disparity,
                          &stereo_score)) {
      continue;
    }

    Eigen::Vector3d p_cam0;
    if (!TriangulateRectified(corner.x, corner.y, disparity, rectification_.fx, rectification_.fy,
                              rectification_.cx, rectification_.cy, rectification_.baseline_m,
                              static_cast<double>(match_options_.stereo_min_disparity), &p_cam0)) {
      continue;
    }

    added.push_back({corner.x, corner.y, p_cam0});
    occupied.insert(cell);
  }

  return added;
}

StereoVoFrontend::FrameResult StereoVoFrontend::ProcessFrame(const GrayImage& cam0_raw,
                                                              const GrayImage& cam1_raw) {
  const GrayImage cam0_rect = rectification_.Rectify(cam0_raw, 0);
  const GrayImage cam1_rect = rectification_.Rectify(cam1_raw, 1);

  if (!have_prev_frame_) {
    tracked_points_ = DetectAndTriangulateNew(cam0_rect, cam1_rect, {});
    prev_cam0_rectified_ = cam0_rect;
    have_prev_frame_ = true;
    return {false, Sophus::SE3d(), static_cast<int>(tracked_points_.size()), 0};
  }

  const int num_tracked_into_frame = static_cast<int>(tracked_points_.size());

  // Candidates: previous-frame 3D point (dst) paired with the same physical
  // point re-triangulated at the current frame (src), for every point that
  // survives temporal tracking, restereo matching, and triangulation.
  std::vector<Eigen::Vector3d> src_curr, dst_prev;
  std::vector<TrackedPoint> candidates;  // current-frame pixel/3D, aligned with src_curr
  for (const TrackedPoint& prev_point : tracked_points_) {
    double u_curr = 0, v_curr = 0, temporal_score = 0;
    if (!MatchTemporalPatch(prev_cam0_rectified_, cam0_rect, prev_point.u_rect, prev_point.v_rect,
                            match_options_, &u_curr, &v_curr, &temporal_score)) {
      continue;
    }

    double disparity = 0, stereo_score = 0;
    if (!MatchStereoPatch(cam0_rect, cam1_rect, u_curr, v_curr, match_options_, &disparity,
                          &stereo_score)) {
      continue;
    }

    Eigen::Vector3d p_cam0_curr;
    if (!TriangulateRectified(u_curr, v_curr, disparity, rectification_.fx, rectification_.fy,
                              rectification_.cx, rectification_.cy, rectification_.baseline_m,
                              static_cast<double>(match_options_.stereo_min_disparity),
                              &p_cam0_curr)) {
      continue;
    }

    src_curr.push_back(p_cam0_curr);
    dst_prev.push_back(prev_point.p_cam0);
    candidates.push_back({u_curr, v_curr, p_cam0_curr});
  }

  RigidTransform relative;
  std::vector<int> inlier_indices;
  const bool ransac_ok =
      RansacRigidRegistration(src_curr, dst_prev, ransac_options_, &relative, &inlier_indices);

  FrameResult result;
  result.num_tracked = num_tracked_into_frame;
  result.num_inliers = static_cast<int>(inlier_indices.size());
  if (ransac_ok) {
    // relative.R,t map current-cam points into previous-cam coordinates:
    // T_rectcam0prev_rectcam0curr.
    const Sophus::SE3d T_rectcam0prev_rectcam0curr(Sophus::SO3d::fitToSO3(relative.R), relative.t);
    result.has_relative_pose = true;
    result.T_prevbody_currbody = rectification_.T_body_rectcam0 * T_rectcam0prev_rectcam0curr *
                                 rectification_.T_body_rectcam0.inverse();
  }

  // Keep only inlier-surviving points for the next frame's tracking.
  std::vector<TrackedPoint> surviving;
  surviving.reserve(inlier_indices.size());
  for (int idx : inlier_indices) surviving.push_back(candidates[idx]);

  const std::vector<TrackedPoint> new_points = DetectAndTriangulateNew(cam0_rect, cam1_rect, surviving);
  surviving.insert(surviving.end(), new_points.begin(), new_points.end());
  tracked_points_ = std::move(surviving);
  prev_cam0_rectified_ = cam0_rect;

  return result;
}

}  // namespace vio

#include "mono_feature_tracker.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

#include "stereo_rectifier.h"

namespace vio {

Eigen::Vector2d UndistortRadTan(const Eigen::Vector2d& distorted, const RadTanDistortion& d) {
  // x_d = x * radial(x) + tangential(x)  =>  x = (x_d - tangential(x)) / radial(x).
  Eigen::Vector2d x = distorted;
  for (int iter = 0; iter < 20; ++iter) {
    const double r2 = x.squaredNorm();
    const double radial = 1.0 + d.k1 * r2 + d.k2 * r2 * r2;
    const Eigen::Vector2d tangential(2.0 * d.p1 * x.x() * x.y() + d.p2 * (r2 + 2.0 * x.x() * x.x()),
                                     d.p1 * (r2 + 2.0 * x.y() * x.y()) + 2.0 * d.p2 * x.x() * x.y());
    x = (distorted - tangential) / radial;
  }
  return x;
}

MonoFeatureTracker::MonoFeatureTracker(const CameraCalibration& camera,
                                       const MonoTrackerOptions& options)
    : camera_(camera), options_(options) {}

Eigen::Vector3d MonoFeatureTracker::PixelToBearing(double u, double v) const {
  const CameraIntrinsics& k = camera_.intrinsics;
  const Eigen::Vector2d x =
      UndistortRadTan(Eigen::Vector2d((u - k.cu) / k.fu, (v - k.cv) / k.fv), camera_.distortion);
  return Eigen::Vector3d(x.x(), x.y(), 1.0).normalized();
}

bool MonoFeatureTracker::BearingToPixel(const Eigen::Vector3d& bearing, double* u,
                                        double* v) const {
  if (bearing.z() <= 1e-6) return false;
  const Eigen::Vector2d xd =
      DistortRadTan(Eigen::Vector2d(bearing.x() / bearing.z(), bearing.y() / bearing.z()),
                    camera_.distortion);
  *u = camera_.intrinsics.fu * xd.x() + camera_.intrinsics.cu;
  *v = camera_.intrinsics.fv * xd.y() + camera_.intrinsics.cv;
  return true;
}

bool MonoFeatureTracker::InsideBorder(double u, double v) const {
  const int b = options_.border_px;
  return u >= b && v >= b && u <= camera_.width - 1 - b && v <= camera_.height - 1 - b;
}

std::vector<TrackedFeature> MonoFeatureTracker::Track(const GrayImage& image,
                                                      const Sophus::SO3d& R_prev_curr) {
  std::vector<TrackedFeature> tracked;
  if (have_prev_) {
    const Sophus::SO3d R_curr_prev = R_prev_curr.inverse();
    for (const TrackedFeature& f : features_) {
      // Rotation-only prediction: exact for distant points, and translation
      // between 20 Hz frames is a few pixels at most for EuRoC's scenes.
      double u_guess = f.u, v_guess = f.v;
      if (!BearingToPixel(R_curr_prev * f.bearing, &u_guess, &v_guess) ||
          !InsideBorder(u_guess, v_guess)) {
        continue;
      }
      u_guess = std::round(u_guess);
      v_guess = std::round(v_guess);

      double u = 0, v = 0, score = 0;
      if (!MatchTemporalPatchNear(prev_image_, image, f.u, f.v, u_guess, v_guess,
                                  options_.search_radius, options_.match, &u, &v, &score)) {
        continue;
      }
      if (!InsideBorder(u, v)) continue;

      // Forward-backward check: track the match back into the previous
      // image with an equally wide search, centred where undoing the
      // forward displacement would land. A wrong forward match rarely maps
      // back onto its starting point.
      double u_back = 0, v_back = 0;
      if (!MatchTemporalPatchNear(image, prev_image_, u, v, std::round(f.u + (u_guess - u)),
                                  std::round(f.v + (v_guess - v)), options_.search_radius,
                                  options_.match, &u_back, &v_back, &score)) {
        continue;
      }
      if (std::hypot(u_back - f.u, v_back - f.v) > options_.forward_backward_threshold_px) {
        continue;
      }
      tracked.push_back({f.id, u, v, PixelToBearing(u, v)});
    }
  }

  // Top up with fresh Harris corners in grid cells no tracked feature
  // occupies, strongest first.
  if (static_cast<int>(tracked.size()) < options_.max_features) {
    const int cell = std::max(1, options_.harris.cell_size);
    auto cell_of = [cell](double u, double v) {
      return std::make_pair(static_cast<int>(v) / cell, static_cast<int>(u) / cell);
    };
    std::set<std::pair<int, int>> occupied;
    for (const TrackedFeature& f : tracked) occupied.insert(cell_of(f.u, f.v));

    std::vector<Corner> corners = DetectHarrisCorners(image, options_.harris);
    std::sort(corners.begin(), corners.end(),
              [](const Corner& a, const Corner& b) { return a.score > b.score; });
    for (const Corner& c : corners) {
      if (static_cast<int>(tracked.size()) >= options_.max_features) break;
      if (!InsideBorder(c.x, c.y)) continue;
      const auto key = cell_of(c.x, c.y);
      if (occupied.count(key)) continue;
      occupied.insert(key);
      tracked.push_back({next_id_++, c.x, c.y, PixelToBearing(c.x, c.y)});
    }
  }

  features_ = tracked;
  prev_image_ = image;
  have_prev_ = true;
  return tracked;
}

}  // namespace vio

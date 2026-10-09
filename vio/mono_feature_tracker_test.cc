#include "mono_feature_tracker.h"

#include <cmath>
#include <map>
#include <random>

#include <gtest/gtest.h>

#include "stereo_rectifier.h"

namespace vio {
namespace {

// A smooth, non-periodic texture: a sum of Gaussian blobs evaluated
// analytically, so a sub-pixel-shifted copy is exact (no resampling error).
class BlobTexture {
 public:
  explicit BlobTexture(int count, double extent, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> pos(-40.0, extent + 40.0);
    std::uniform_real_distribution<double> amp(-90.0, 90.0);
    std::uniform_real_distribution<double> sigma(2.0, 5.0);
    for (int i = 0; i < count; ++i) blobs_.push_back({pos(rng), pos(rng), amp(rng), sigma(rng)});
  }

  // The image whose pixel (u, v) shows the texture at (u - du, v - dv): the
  // scene moved by (+du, +dv).
  GrayImage Render(int cols, int rows, double du, double dv) const {
    GrayImage img;
    img.cols = cols;
    img.rows = rows;
    img.pixels.resize(static_cast<size_t>(cols) * rows);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        double value = 128.0;
        for (const Blob& b : blobs_) {
          const double dx = c - du - b.x, dy = r - dv - b.y;
          value += b.a * std::exp(-(dx * dx + dy * dy) / (2 * b.s * b.s));
        }
        img.pixels[static_cast<size_t>(r) * cols + c] =
            static_cast<uint8_t>(std::lround(std::min(255.0, std::max(0.0, value))));
      }
    }
    return img;
  }

 private:
  struct Blob {
    double x, y, a, s;
  };
  std::vector<Blob> blobs_;
};

CameraCalibration PinholeCamera(int cols, int rows) {
  CameraCalibration cam;
  cam.width = cols;
  cam.height = rows;
  cam.intrinsics = {200.0, 200.0, cols / 2.0, rows / 2.0};
  return cam;
}

TEST(UndistortRadTan, InvertsDistortRadTan) {
  // EuRoC cam0 coefficients.
  const RadTanDistortion d{-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05};
  for (double x = -0.6; x <= 0.6; x += 0.15) {
    for (double y = -0.4; y <= 0.4; y += 0.1) {
      const Eigen::Vector2d p(x, y);
      EXPECT_LT((UndistortRadTan(DistortRadTan(p, d), d) - p).norm(), 1e-9);
    }
  }
}

TEST(MonoFeatureTracker, BearingPixelRoundTrip) {
  // EuRoC cam0, including the image corners where distortion is strongest.
  CameraCalibration cam;
  cam.width = 752;
  cam.height = 480;
  cam.intrinsics = {458.654, 457.296, 367.215, 248.375};
  cam.distortion = {-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05};
  const MonoFeatureTracker tracker(cam, MonoTrackerOptions());
  for (const auto& [u, v] : std::vector<std::pair<double, double>>{{0, 0}, {376, 240}, {751, 479}, {0, 479}}) {
    double u2 = 0, v2 = 0;
    ASSERT_TRUE(tracker.BearingToPixel(tracker.PixelToBearing(u, v), &u2, &v2));
    EXPECT_NEAR(u2, u, 1e-6);
    EXPECT_NEAR(v2, v, 1e-6);
  }
}

TEST(MatchTemporalPatchNear, SubpixelRefinementRecoversFractionalShift) {
  const BlobTexture texture(400, 200, 1);
  const GrayImage prev = texture.Render(200, 160, 0, 0);
  const GrayImage curr = texture.Render(200, 160, 2.3, -1.6);
  auto mean_error = [&](bool subpixel) {
    PatchMatchOptions options;
    options.temporal_subpixel = subpixel;
    int matched = 0;
    double err_sum = 0;
    for (const Corner& c : DetectHarrisCorners(prev, HarrisOptions())) {
      // Near the edge the true match can fall outside the image.
      if (c.x < 10 || c.y < 10 || c.x > prev.cols - 11 || c.y > prev.rows - 11) continue;
      double uc = 0, vc = 0, score = 0;
      if (!MatchTemporalPatchNear(prev, curr, c.x, c.y, c.x, c.y, 6, options, &uc, &vc, &score)) {
        continue;
      }
      err_sum += std::hypot(uc - (c.x + 2.3), vc - (c.y - 1.6));
      ++matched;
    }
    EXPECT_GT(matched, 10);
    return err_sum / matched;
  };
  const double integer_error = mean_error(false);  // ~0.5 for a (0.3, 0.4) fraction
  const double subpixel_error = mean_error(true);
  EXPECT_LT(subpixel_error, 0.5 * integer_error);
  EXPECT_LT(subpixel_error, 0.2);
}

TEST(MonoFeatureTracker, KeepsIdsAcrossFrames) {
  const BlobTexture texture(400, 200, 2);
  MonoFeatureTracker tracker(PinholeCamera(200, 160), MonoTrackerOptions());
  const std::vector<TrackedFeature> first =
      tracker.Track(texture.Render(200, 160, 0, 0), Sophus::SO3d());
  ASSERT_GT(first.size(), 8u);
  const std::vector<TrackedFeature> second =
      tracker.Track(texture.Render(200, 160, 3.4, 1.7), Sophus::SO3d());

  std::map<int, TrackedFeature> before;
  for (const TrackedFeature& f : first) before[f.id] = f;
  int kept = 0;
  for (const TrackedFeature& f : second) {
    const auto it = before.find(f.id);
    if (it == before.end()) continue;
    ++kept;
    EXPECT_NEAR(f.u, it->second.u + 3.4, 0.35);
    EXPECT_NEAR(f.v, it->second.v + 1.7, 0.35);
  }
  EXPECT_GE(kept, static_cast<int>(first.size()) * 3 / 4);
}

TEST(MonoFeatureTracker, RotationPriorTracksMotionBeyondSearchWindow) {
  // A 25 px shift is twice the default search radius. A camera yaw of
  // theta about +y moves image content by about -f * theta in u; feed the
  // tracker the rotation that predicts the shift.
  const BlobTexture texture(400, 200, 3);
  const GrayImage frame0 = texture.Render(200, 160, 0, 0);
  const GrayImage frame1 = texture.Render(200, 160, 25, 0);
  const Sophus::SO3d R_prev_curr = Sophus::SO3d::exp(Eigen::Vector3d(0, -25.0 / 200.0, 0));

  auto count_kept = [&](const Sophus::SO3d& prior) {
    MonoFeatureTracker tracker(PinholeCamera(200, 160), MonoTrackerOptions());
    std::map<int, bool> ids;
    for (const TrackedFeature& f : tracker.Track(frame0, Sophus::SO3d())) ids[f.id] = true;
    int kept = 0;
    for (const TrackedFeature& f : tracker.Track(frame1, prior)) kept += ids.count(f.id);
    return kept;
  };
  // Without the prior, the true match is outside the window. A few features
  // still pair up with a look-alike blob in both directions (that is what
  // EqvioParams::outlier_chi2_threshold is for), but most are lost.
  const int without_prior = count_kept(Sophus::SO3d());
  const int with_prior = count_kept(R_prev_curr);
  EXPECT_GE(with_prior, 12);
  EXPECT_LT(3 * without_prior, with_prior);
}

}  // namespace
}  // namespace vio

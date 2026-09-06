#include "patch_matcher.h"

#include "gtest/gtest.h"

namespace vio {
namespace {

// A single distinctive 7x7 block (asymmetric under any shift, so no wrong
// candidate offset can coincidentally reproduce it) on an otherwise flat
// background. This makes the true offset the unique, unambiguous SSD
// minimum: any wrong candidate compares the block against flat background
// and scores far worse, rather than merely "a bit worse".
GrayImage MakeBackgroundWithBlock(int rows, int cols, int block_u, int block_v, int radius) {
  GrayImage image;
  image.rows = rows;
  image.cols = cols;
  image.pixels.assign(static_cast<size_t>(rows) * cols, 0);
  for (int dr = -radius; dr <= radius; ++dr) {
    for (int dc = -radius; dc <= radius; ++dc) {
      const int r = block_v + dr;
      const int c = block_u + dc;
      const uint8_t value = static_cast<uint8_t>(((dr + radius) * 37 + (dc + radius) * 53) % 250 + 5);
      image.pixels[static_cast<size_t>(r) * cols + c] = value;
    }
  }
  return image;
}

}  // namespace

TEST(PatchMatcher, MatchStereoPatchRecoversKnownDisparity) {
  const int radius = 3;
  const GrayImage left = MakeBackgroundWithBlock(200, 200, 100, 100, radius);
  // Place the same block at u=85 in "right", i.e. disparity = 100-85 = 15.
  const GrayImage right = MakeBackgroundWithBlock(200, 200, 85, 100, radius);

  PatchMatchOptions options;
  options.patch_radius = radius;
  double disparity = 0, score = 0;
  ASSERT_TRUE(MatchStereoPatch(left, right, 100, 100, options, &disparity, &score));
  EXPECT_NEAR(disparity, 15.0, 1e-9);
  EXPECT_LT(score, 1e-6);
}

TEST(PatchMatcher, MatchTemporalPatchRecoversKnownShift) {
  const int radius = 3;
  const GrayImage prev = MakeBackgroundWithBlock(200, 200, 100, 100, radius);
  // Same block moved by (du,dv) = (-8, 5) in "curr".
  const GrayImage curr = MakeBackgroundWithBlock(200, 200, 92, 105, radius);

  PatchMatchOptions options;
  options.patch_radius = radius;
  double u_curr = 0, v_curr = 0, score = 0;
  ASSERT_TRUE(MatchTemporalPatch(prev, curr, 100, 100, options, &u_curr, &v_curr, &score));
  EXPECT_NEAR(u_curr, 92.0, 1e-9);
  EXPECT_NEAR(v_curr, 105.0, 1e-9);
  EXPECT_LT(score, 1e-6);
}

TEST(PatchMatcher, MatchFailsOnFeaturelessRegion) {
  GrayImage a, b;
  a.rows = b.rows = 100;
  a.cols = b.cols = 100;
  a.pixels.assign(100 * 100, 100);
  b.pixels.assign(100 * 100, 100);

  PatchMatchOptions options;
  double disparity = 0, score = 0;
  EXPECT_FALSE(MatchStereoPatch(a, b, 50, 50, options, &disparity, &score));

  double u_curr = 0, v_curr = 0;
  EXPECT_FALSE(MatchTemporalPatch(a, b, 50, 50, options, &u_curr, &v_curr, &score));
}

}  // namespace vio

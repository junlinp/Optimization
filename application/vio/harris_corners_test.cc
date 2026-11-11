#include "harris_corners.h"

#include <cmath>

#include "gtest/gtest.h"

namespace vio {
namespace {

GrayImage MakeSquareImage() {
  GrayImage image;
  image.rows = 100;
  image.cols = 100;
  image.pixels.assign(100 * 100, 0);
  for (int r = 30; r <= 70; ++r) {
    for (int c = 30; c <= 70; ++c) {
      image.pixels[r * 100 + c] = 255;
    }
  }
  return image;
}

bool HasCornerNear(const std::vector<Corner>& corners, double x, double y, double tol) {
  for (const Corner& corner : corners) {
    if (std::hypot(corner.x - x, corner.y - y) <= tol) return true;
  }
  return false;
}

}  // namespace

TEST(HarrisCorners, DetectsCornersOfASyntheticSquare) {
  const GrayImage image = MakeSquareImage();

  HarrisOptions options;
  options.cell_size = 20;
  const std::vector<Corner> corners = DetectHarrisCorners(image, options);

  EXPECT_TRUE(HasCornerNear(corners, 30, 30, 2.0));
  EXPECT_TRUE(HasCornerNear(corners, 70, 30, 2.0));
  EXPECT_TRUE(HasCornerNear(corners, 30, 70, 2.0));
  EXPECT_TRUE(HasCornerNear(corners, 70, 70, 2.0));
}

TEST(HarrisCorners, FlatImageProducesNoCorners) {
  GrayImage image;
  image.rows = 50;
  image.cols = 50;
  image.pixels.assign(50 * 50, 128);

  const std::vector<Corner> corners = DetectHarrisCorners(image, HarrisOptions());
  EXPECT_TRUE(corners.empty());
}

}  // namespace vio

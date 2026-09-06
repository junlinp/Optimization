#include "gtest/gtest.h"
#include "sophus/se3.hpp"
#include "stereo_rectifier.h"
#include "triangulation.h"

namespace vio {
namespace {

CameraCalibration MakeCalibration(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                                  double fu, double fv, double cu, double cv) {
  CameraCalibration calib;
  calib.T_BS = Sophus::SE3d(Sophus::SO3d::fitToSO3(R), t);
  calib.intrinsics = {fu, fv, cu, cv};
  calib.distortion = {0, 0, 0, 0};
  calib.width = 640;
  calib.height = 480;
  calib.rate_hz = 20;
  return calib;
}

}  // namespace

TEST(Triangulation, TriangulateRectifiedRecoversKnownPoint) {
  const double fx = 400, fy = 400, cx = 320, cy = 240, baseline = 0.1;
  const Eigen::Vector3d truth(0.5, -0.2, 2.0);

  const double u_left = fx * truth.x() / truth.z() + cx;
  const double v_left = fy * truth.y() / truth.z() + cy;
  const double disparity = fx * baseline / truth.z();

  Eigen::Vector3d recovered;
  ASSERT_TRUE(TriangulateRectified(u_left, v_left, disparity, fx, fy, cx, cy, baseline, 0.0,
                                   &recovered));
  EXPECT_LT((recovered - truth).norm(), 1e-9);
}

TEST(Triangulation, TriangulateRectifiedRejectsNonPositiveDisparity) {
  Eigen::Vector3d point;
  EXPECT_FALSE(TriangulateRectified(320, 240, 0.0, 400, 400, 320, 240, 0.1, 0.0, &point));
}

TEST(StereoRectifier, DistortRadTanMatchesHandComputedValue) {
  RadTanDistortion d{0.01, 0.002, 0.001, -0.0005};
  const Eigen::Vector2d result = DistortRadTan(Eigen::Vector2d(0.1, -0.05), d);
  EXPECT_NEAR(result.x(), 0.09998628125000002, 1e-15);
  EXPECT_NEAR(result.y(), -0.04998376562500001, 1e-15);
}

TEST(StereoRectifier, ComputeStereoRectificationIdentityCaseIsNearIdentityRemap) {
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const CameraCalibration cam0 = MakeCalibration(I, Eigen::Vector3d(0, 0, 0), 400, 400, 200, 150);
  const CameraCalibration cam1 =
      MakeCalibration(I, Eigen::Vector3d(0.1, 0, 0), 400, 400, 200, 150);

  const StereoRectification rect = ComputeStereoRectification(cam0, cam1, 400, 400);

  EXPECT_NEAR(rect.fx, 400.0, 1e-9);
  EXPECT_NEAR(rect.baseline_m, 0.1, 1e-9);

  // With identical intrinsics, zero distortion, and a baseline exactly along
  // cam0's own x-axis, the rectifying rotation for cam0 is the identity, so
  // its remap should be the identity mapping: a bright pixel round-trips to
  // (approximately) the same location.
  GrayImage raw;
  raw.rows = 400;
  raw.cols = 400;
  raw.pixels.assign(400 * 400, 0);
  raw.pixels[150 * 400 + 200] = 255;  // bright pixel at (u=200, v=150)

  const GrayImage rectified = rect.Rectify(raw, 0);

  int best_r = -1, best_c = -1, best_val = -1;
  for (int r = 0; r < rectified.rows; ++r) {
    for (int c = 0; c < rectified.cols; ++c) {
      if (rectified.at(r, c) > best_val) {
        best_val = rectified.at(r, c);
        best_r = r;
        best_c = c;
      }
    }
  }
  EXPECT_LE(std::abs(best_r - 150), 1);
  EXPECT_LE(std::abs(best_c - 200), 1);
}

}  // namespace vio

#include "rigid_registration.h"

#include <algorithm>

#include "gtest/gtest.h"
#include "sophus/so3.hpp"

namespace vio {
namespace {

// 20 non-planar, non-collinear points (spans a full-rank covariance), no RNG.
std::vector<Eigen::Vector3d> SamplePoints() {
  return {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1},
      {2, 0, 0}, {0, 2, 0}, {0, 0, 2}, {2, 1, 0}, {1, 2, 0}, {0, 2, 1}, {2, 0, 1}, {1, 0, 2},
      {0, 1, 2}, {2, 2, 0}, {2, 0, 2}, {0, 2, 2},
  };
}

}  // namespace

TEST(RigidRegistration, UmeyamaRecoversKnownTransformExactly) {
  const Eigen::Matrix3d R_true = Sophus::SO3d::exp(Eigen::Vector3d(0.1, -0.2, 0.3)).matrix();
  const Eigen::Vector3d t_true(1, 2, -0.5);

  const std::vector<Eigen::Vector3d> src = SamplePoints();
  std::vector<Eigen::Vector3d> dst;
  for (const auto& p : src) dst.push_back(R_true * p + t_true);

  const RigidTransform result = UmeyamaAlignment(src, dst);
  EXPECT_LT((result.R - R_true).norm(), 1e-9);
  EXPECT_LT((result.t - t_true).norm(), 1e-9);
}

TEST(RigidRegistration, UmeyamaHandlesReflectionCase) {
  // dst is src mirrored through the z=0 plane: an improper (det=-1)
  // transform relates them, so the naive R=V*U^T fit would itself be a
  // reflection. The sign-corrected algorithm must still return a proper
  // rotation (det=+1) as the closest fit.
  const std::vector<Eigen::Vector3d> src = SamplePoints();
  std::vector<Eigen::Vector3d> dst;
  for (const auto& p : src) dst.push_back(Eigen::Vector3d(p.x(), p.y(), -p.z()));

  const RigidTransform result = UmeyamaAlignment(src, dst);
  EXPECT_NEAR(result.R.determinant(), 1.0, 1e-9);
}

TEST(RigidRegistration, RansacRejectsInjectedOutliers) {
  const Eigen::Matrix3d R_true = Sophus::SO3d::exp(Eigen::Vector3d(0.1, -0.2, 0.3)).matrix();
  const Eigen::Vector3d t_true(1, 2, -0.5);

  const std::vector<Eigen::Vector3d> src = SamplePoints();
  std::vector<Eigen::Vector3d> dst;
  for (const auto& p : src) dst.push_back(R_true * p + t_true);

  const std::vector<int> outlier_indices = {2, 5, 9, 12, 15, 18};  // 6 of 20 = 30%
  for (int idx : outlier_indices) {
    dst[idx] += Eigen::Vector3d(5.0, 5.0, 5.0);  // far beyond inlier_threshold_m
  }

  RansacOptions options;
  RigidTransform result;
  std::vector<int> inlier_indices;
  ASSERT_TRUE(RansacRigidRegistration(src, dst, options, &result, &inlier_indices));

  std::sort(inlier_indices.begin(), inlier_indices.end());
  EXPECT_EQ(inlier_indices.size(), src.size() - outlier_indices.size());
  for (int idx : outlier_indices) {
    EXPECT_FALSE(std::binary_search(inlier_indices.begin(), inlier_indices.end(), idx))
        << "outlier index " << idx << " was kept as an inlier";
  }
  for (size_t i = 0; i < src.size(); ++i) {
    const bool is_outlier =
        std::find(outlier_indices.begin(), outlier_indices.end(), i) != outlier_indices.end();
    if (!is_outlier) {
      EXPECT_TRUE(std::binary_search(inlier_indices.begin(), inlier_indices.end(),
                                     static_cast<int>(i)))
          << "true inlier index " << i << " was dropped";
    }
  }

  EXPECT_LT((result.R - R_true).norm(), 1e-6);
  EXPECT_LT((result.t - t_true).norm(), 1e-6);
}

}  // namespace vio

#include "rigid_registration.h"

#include <algorithm>
#include <numeric>
#include <random>

namespace vio {

RigidTransform UmeyamaAlignment(const std::vector<Eigen::Vector3d>& src,
                                const std::vector<Eigen::Vector3d>& dst) {
  const size_t n = src.size();

  Eigen::Vector3d src_mean = Eigen::Vector3d::Zero();
  Eigen::Vector3d dst_mean = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < n; ++i) {
    src_mean += src[i];
    dst_mean += dst[i];
  }
  src_mean /= static_cast<double>(n);
  dst_mean /= static_cast<double>(n);

  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < n; ++i) {
    H += (src[i] - src_mean) * (dst[i] - dst_mean).transpose();
  }

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Matrix3d& U = svd.matrixU();
  const Eigen::Matrix3d& V = svd.matrixV();

  const double d = (V * U.transpose()).determinant() < 0 ? -1.0 : 1.0;
  Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
  S(2, 2) = d;

  RigidTransform result;
  result.R = V * S * U.transpose();
  result.t = dst_mean - result.R * src_mean;
  return result;
}

bool RansacRigidRegistration(const std::vector<Eigen::Vector3d>& src,
                             const std::vector<Eigen::Vector3d>& dst,
                             const RansacOptions& options, RigidTransform* result,
                             std::vector<int>* inlier_indices) {
  const int n = static_cast<int>(src.size());
  std::vector<int> best_inliers;

  if (n >= options.min_sample_size) {
    std::mt19937 rng(options.random_seed);
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    for (int iter = 0; iter < options.max_iterations; ++iter) {
      std::shuffle(indices.begin(), indices.end(), rng);
      std::vector<Eigen::Vector3d> sample_src, sample_dst;
      for (int i = 0; i < options.min_sample_size; ++i) {
        sample_src.push_back(src[indices[i]]);
        sample_dst.push_back(dst[indices[i]]);
      }
      const RigidTransform candidate = UmeyamaAlignment(sample_src, sample_dst);

      std::vector<int> inliers;
      for (int i = 0; i < n; ++i) {
        const double error = (dst[i] - (candidate.R * src[i] + candidate.t)).norm();
        if (error < options.inlier_threshold_m) inliers.push_back(i);
      }
      if (inliers.size() > best_inliers.size()) best_inliers = std::move(inliers);
    }
  }

  if (!best_inliers.empty()) {
    std::vector<Eigen::Vector3d> inlier_src, inlier_dst;
    inlier_src.reserve(best_inliers.size());
    inlier_dst.reserve(best_inliers.size());
    for (int idx : best_inliers) {
      inlier_src.push_back(src[idx]);
      inlier_dst.push_back(dst[idx]);
    }
    *result = UmeyamaAlignment(inlier_src, inlier_dst);
  } else {
    *result = RigidTransform();
  }
  *inlier_indices = best_inliers;

  return static_cast<int>(best_inliers.size()) >= options.min_inliers;
}

}  // namespace vio

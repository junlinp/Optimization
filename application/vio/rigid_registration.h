#ifndef VIO_RIGID_REGISTRATION_H_
#define VIO_RIGID_REGISTRATION_H_
#include <vector>

#include <Eigen/Dense>

namespace vio {

struct RigidTransform {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

// Closed-form rigid registration (Umeyama/Arun): finds R,t minimizing
// sum ||dst_i - (R*src_i + t)||^2. Requires src.size()==dst.size()>=3.
RigidTransform UmeyamaAlignment(const std::vector<Eigen::Vector3d>& src,
                                const std::vector<Eigen::Vector3d>& dst);

struct RansacOptions {
  int max_iterations = 200;
  double inlier_threshold_m = 0.05;
  int min_sample_size = 3;
  int min_inliers = 8;
  unsigned random_seed = 12345;  // fixed for deterministic tests
};

// Returns false if fewer than min_inliers inliers are found after all
// iterations (result/inlier_indices are still populated with the best
// attempt found, for diagnostics).
bool RansacRigidRegistration(const std::vector<Eigen::Vector3d>& src,
                             const std::vector<Eigen::Vector3d>& dst,
                             const RansacOptions& options, RigidTransform* result,
                             std::vector<int>* inlier_indices);

}  // namespace vio
#endif  // VIO_RIGID_REGISTRATION_H_

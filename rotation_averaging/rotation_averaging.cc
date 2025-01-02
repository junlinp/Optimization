#include "rotation_averaging/rotation_averaging.h"

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <ceres/rotation.h>

namespace {
// Cost function for the rotation error
struct RotationError {
  RotationError(const Eigen::Quaterniond &relative_rotation)
      : relative_rotation(relative_rotation) {}

  template <typename T>
  bool operator()(const T *const rotation_i, const T*const rotation_j, T *residual) const {
    T rotation_matrix_i[9];
    T rotation_matrix_j[9];

    ceres::AngleAxisToRotationMatrix(rotation_i, rotation_matrix_i);
    ceres::AngleAxisToRotationMatrix(rotation_j, rotation_matrix_j);

    Eigen::Map<Eigen::Matrix<T,3, 3>> eigen_rotation_i(rotation_matrix_i);
    Eigen::Map<Eigen::Matrix<T,3,3>> eigen_rotation_j(rotation_matrix_j);
    Eigen::Map<Eigen::Matrix<T,3,3>> eigen_rotation_residual(residual);
    eigen_rotation_residual = eigen_rotation_j.transpose() * relative_rotation.toRotationMatrix().cast<T>() * eigen_rotation_i;
    return true;
  }
  Eigen::Quaterniond relative_rotation;
};

} // namespace

Eigen::Quaterniond RotationAveraging::averageRotations(
    const std::map<std::pair<size_t,size_t>, Eigen::Quaterniond> &relative_rotations,
    size_t num_of_rotations, std::vector<Eigen::Quaterniond> *rotations) {

  // Create a Ceres problem
  ceres::Problem problem;


  std::vector<Eigen::Vector3d> rotation_parameters;
  for (int i = 0; i < num_of_rotations; i++) {
    rotation_parameters.push_back(Eigen::Vector3d::Random());
  }

  // Add residuals for each pair of rotations
  for (const auto &[key, relative_rotation] : relative_rotations) {
    const int i = key.first;
    const int j = key.second;


    auto cost_function = new ceres::AutoDiffCostFunction<RotationError, 9, 3, 3>(
										 new RotationError(relative_rotation));
    // Create a residual block for the rotation error
    problem.AddResidualBlock(cost_function,
			     nullptr, rotation_parameters[i].data(), rotation_parameters[j].data());
  }
  std::cout << "Solve" << std::endl;
  // Set the solver options
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;

  // Solve the problem
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;

  rotations->clear();
  for (int i = 0; i < num_of_rotations;i++) {
    double q[4];
    ceres::AngleAxisToQuaternion(rotation_parameters[i].data(), q);
    Eigen::Quaterniond r(q[0], q[1], q[2],q[3]);
    rotations->push_back(r);
  }
  // Return the average rotation as a quaternion
  return Eigen::Quaterniond::Identity();
}

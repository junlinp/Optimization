#include "rotation_averaging/rotation_averaging.h"

#include <ceres/ceres.h>
#include <Eigen/Dense>

Eigen::Quaterniond RotationAveraging::averageRotations(
    const std::map<std::pair<int, int>, Eigen::Quaterniond> &relative_rotations,
    size_t num_of_rotations, std::vector<Eigen::Quaterniond> *rotations) {

  // Create a Ceres problem
  ceres::Problem problem;

  // Create a variable to hold the average rotation
  Eigen::Quaterniond average_rotation = Eigen::Quaterniond::Identity();
  double average_rotation_params[4] = {
      average_rotation.w(), average_rotation.x(), average_rotation.y(),
      average_rotation.z()};

  // Add residuals for each pair of rotations
  for (const auto &[key, relative_rotation] : rotations) {
    const int i = key.first;
    const int j = key.second;

    // Create a residual block for the rotation error
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<RotationError, 4, 4>(
            new RotationError(relative_rotation)),
        nullptr, average_rotation_params);
  }

  // Set the solver options
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  // Solve the problem
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Return the average rotation as a quaternion
  return Eigen::Quaterniond(
      average_rotation_params[0], average_rotation_params[1],
      average_rotation_params[2], average_rotation_params[3]);
}

// Cost function for the rotation error
struct RotationError {
    RotationError(const Eigen::Quaterniond& relative_rotation) : relative_rotation(relative_rotation) {}

    template <typename T>
    bool operator()(const T* const average_rotation, T* residual) const {
        Eigen::Quaternion<T> avg_rotation(average_rotation[0], average_rotation[1], average_rotation[2], average_rotation[3]);
        Eigen::Quaternion<T> error = relative_rotation.cast<T>() * avg_rotation.inverse();
        residual[0] = error.x();
        residual[1] = error.y();
        residual[2] = error.z();
        residual[3] = error.w();
        return true;
    }

    Eigen::Quaterniond relative_rotation;
}

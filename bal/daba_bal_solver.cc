#include "daba_bal_solver.h"
#include "ceres/problem.h"
#include "ceres/rotation.h"
#include "ceres/solver.h"

template<class T>
Eigen::Matrix<T, 3, 1> UndistortionRay(const T* focal_and_distortion, T u, T v) {
  const T &focal = focal_and_distortion[0];
  const T &k1 = focal_and_distortion[1];
  const T &k2 = focal_and_distortion[2];
  T l2_norm_uv = u * u + v * v;
  return Eigen::Matrix<T, 3, 1> (u / focal, v / focal,
                             T(1.0) + k1 * l2_norm_uv +
                                 k2 * l2_norm_uv * l2_norm_uv);
}

template<class T>
T MinimumDepth(const T* camera_parameters, const T* landmark_position, T u, T v) {
  using Tvector3d = Eigen::Matrix<T, 3, 1>;
  Tvector3d pij = UndistortionRay(camera_parameters + 6, u, v);
  Eigen::Map<const Tvector3d> ti(camera_parameters +  3);
  Eigen::Map<const Tvector3d> pj(landmark_position);
  T rotation_output[3];
  ceres::AngleAxisRotatePoint(camera_parameters, pij.data(), rotation_output);
  Eigen::Map<Tvector3d> rotation_output_map(rotation_output);
  return (pj - ti).dot(rotation_output_map) / (pj - ti).dot(pj - ti);
}

template <class T>
void IntermediateTerm(const T *camera_parameters, const T *position,
                      T *residual, const T &u, const T &v) {
  using Tvector3d = Eigen::Matrix<T, 3, 1>;
  Tvector3d pij = UndistortionRay(camera_parameters + 6, u, v);

  Eigen::Map<const Tvector3d> ti(camera_parameters + 3);
  Eigen::Map<const Tvector3d> pj(position);
  ceres::AngleAxisRotatePoint(camera_parameters, pij.data(), residual);
  Eigen::Map< Tvector3d> residual_map(residual);
  T lambda = (pj - ti).dot(residual_map) / (pj - ti).dot(pj - ti);
  residual_map += lambda * (ti + pj);
}

struct CameraSurrogateCostFunction {
  double u_, v_;
  CameraSurrogateCostFunction(double u, double v) : u_(u), v_(v) {}

  template <class T>
  bool operator()(const T *camera_parameters,
                  const T *condition_camera_parameters,
                  const T *condition_landmark_parameters, T *residual) const {
    using Tvector3d = Eigen::Matrix<T, 3, 1>;
    Tvector3d pij = UndistortionRay<T>(camera_parameters + 6, T(u_), T(v_));
    ceres::AngleAxisRotatePoint(camera_parameters, pij.data(), residual);

    Eigen::Map<const Tvector3d> ti(camera_parameters + 3);
    T lambda_k = MinimumDepth(condition_camera_parameters,
                              condition_landmark_parameters, T(u_), T(v_));
    Tvector3d gij_k;
    IntermediateTerm(condition_camera_parameters, condition_landmark_parameters, gij_k.data(), T(u_), T(v_));
    Eigen::Map<Tvector3d> residual_map(residual);
    residual_map += lambda_k * ti - gij_k;
    return true;
  }
};

struct LandmarkSurrogatecostFunction {
    double u_, v_;
    LandmarkSurrogatecostFunction(double u, double v) : u_(u), v_(v) {}

    template <class T>
  bool operator()(const T *landmark_position,
                  const T *condition_camera_parameters,
                  const T *condition_landmark_parameters, T *residual) const {
    using Tvector3d = Eigen::Matrix<T, 3, 1>;
    Eigen::Map<const Tvector3d> pj(landmark_position);
    T lambda_k = MinimumDepth(condition_camera_parameters,
                              condition_landmark_parameters, T(u_), T(v_));
    Tvector3d gij_k;
    IntermediateTerm(condition_camera_parameters, condition_landmark_parameters, gij_k.data(), T(u_), T(v_));
    Eigen::Map<Tvector3d> residual_map(residual);
    residual_map = lambda_k * pj - gij_k;
    return true;
  }
};

void DABAProblemSolver::Solve(Problem &problem) {
  for (auto &&[index, camera_parameter] : problem.cameras_) {
    std::copy(camera_parameter.data(), camera_parameter.data() + 9,
              camera_parameters_[index].begin());
    std::copy(camera_parameter.data(), camera_parameter.data() + 9,
              last_camera_parameters_[index].begin());
  }

  for (auto &&[index, landmark] : problem.points_) {
    std::copy(landmark.data(), landmark.data() + 3,
              landmark_position_[index].begin());
    std::copy(landmark.data(), landmark.data() + 3,
              last_landmark_position_[index].begin());
  }

  std::map<int64_t, ceres::Problem> camera_problems;
  std::map<int64_t, ceres::Problem> landmark_problems;

  for (auto [index_pair, uv] : problem.observations_) {
    int64_t camera_index = index_pair.first;
    int64_t landmark_index = index_pair.second;
    auto &camera_problem = camera_problems[camera_index];
    camera_problem.AddParameterBlock(camera_parameters_[camera_index].data(),
                                     9);
    camera_problem.AddParameterBlock(
        last_camera_parameters_[camera_index].data(), 9);
    camera_problem.SetParameterBlockConstant(
        last_camera_parameters_[camera_index].data());
    camera_problem.AddParameterBlock(
        last_landmark_position_[landmark_index].data(), 3);
    camera_problem.SetParameterBlockConstant(
        last_landmark_position_[landmark_index].data());
    // add cost function
    ceres::CostFunction *camera_cost_function =
        new ceres::AutoDiffCostFunction<CameraSurrogateCostFunction, 3, 9, 9,
                                        3>(
            new CameraSurrogateCostFunction(uv.u(), uv.v()));
    camera_problem.AddResidualBlock(
        camera_cost_function, nullptr, camera_parameters_[camera_index].data(),
        last_camera_parameters_[camera_index].data(),
        last_landmark_position_[landmark_index].data());

    auto &landmark_problem = landmark_problems[landmark_index];
    landmark_problem.AddParameterBlock(
        landmark_position_[landmark_index].data(), 3);
    landmark_problem.AddParameterBlock(
        last_camera_parameters_[camera_index].data(), 9);
    landmark_problem.SetParameterBlockConstant(
        last_camera_parameters_[camera_index].data());
    landmark_problem.AddParameterBlock(
        last_landmark_position_[camera_index].data(), 3);
    landmark_problem.SetParameterBlockConstant(
        last_landmark_position_[camera_index].data());
    ceres::CostFunction *landmark_cost_function = new ceres::AutoDiffCostFunction<LandmarkSurrogatecostFunction, 3, 3, 9, 3>(
        new LandmarkSurrogatecostFunction(uv.u(), uv.v())
    );
    landmark_problem.AddResidualBlock(
        landmark_cost_function, nullptr,
        landmark_position_[landmark_index].data(),
        last_camera_parameters_[camera_index].data(),
        last_landmark_position_[camera_index].data());
  }
  
  for (auto& [_, problem] :camera_problems) {
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
  }

  for (auto& [_, problem] : landmark_problems) {
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
  }

  for (auto &[index, camera_parameter] : problem.cameras_) {
    std::copy(camera_parameters_[index].begin(),
              camera_parameters_[index].end(), camera_parameter.data());
  }

  for (auto &[index, landmark] : problem.points_) {
    std::copy(landmark_position_[index].begin(),
              landmark_position_[index].end(), landmark.data());
  }

}
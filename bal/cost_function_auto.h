#ifndef BAL_COST_FUNCTION_AUTO_H_
#define BAL_COST_FUNCTION_AUTO_H_
#include "ceres/autodiff_cost_function.h"
#include "ceres/cost_function.h"
#include "ceres/rotation.h"

struct ProjectFunction {
  ProjectFunction(double u, double v)
      : u_(u), v_(v) {}

  template <class T>
  bool operator()(const T *camera_param, const T *point, T *residual) const {
    T output_point[3];
    ceres::AngleAxisRotatePoint(camera_param, point, output_point);
    output_point[0] += camera_param[3];
    output_point[1] += camera_param[4];
    output_point[2] += camera_param[5];

    output_point[0] /= output_point[2];
    output_point[1] /= output_point[2];
    T focal = T(camera_param[6]);
    T K1 = T(camera_param[7]);
    T K2 = T(camera_param[8]);

    T p_norm_2 =
        output_point[0] * output_point[0] + output_point[1] * output_point[1];
    T distorsion = T(1.0) + K1 * p_norm_2 + K2 * p_norm_2 * p_norm_2;
    residual[0] = T(u_) - focal * distorsion * output_point[0];
    residual[1] = T(v_) - focal * distorsion * output_point[1];

    return true;
  }

  template <typename... ARGS>
  static ceres::CostFunction *CreateCostFunction(ARGS &&...args) {
    return new ceres::AutoDiffCostFunction<ProjectFunction, 2, 9, 3>(
        new ProjectFunction(std::forward<ARGS>(args)...));
  }
  double u_, v_;
};

template<class T>
Eigen::Matrix<T, 3, 1> UndistortionRay(const T* focal_and_distortion, T u, T v) {
  const T &focal = focal_and_distortion[0];
  const T &k1 = focal_and_distortion[1];
  const T &k2 = focal_and_distortion[2];
  T l2_norm_uv = u * u + v * v;
  return Eigen::Matrix<T, 3, 1> (u , v ,
                             focal + focal * k1 * l2_norm_uv +
                                 focal * k2 * l2_norm_uv * l2_norm_uv);
}

template<class T>
T MinimumDepth(const T* camera_parameters, const T* landmark_position, T u, T v) {
  using Tvector3d = Eigen::Matrix<T, 3, 1>;
  Tvector3d pij = UndistortionRay(camera_parameters + 6, u, v);
  Eigen::Map<const Tvector3d> ti(camera_parameters +  3);
  Eigen::Map<const Tvector3d> pj(landmark_position);
  T rotation_output[3];
  ceres::AngleAxisRotatePoint(camera_parameters, pj.data(), rotation_output);
  for(int i = 0; i < 3; i++) {
    rotation_output[i] += ti(i);
  }
  Eigen::Map<Tvector3d> rotation_output_map(rotation_output);
  return pij.dot(rotation_output_map) / rotation_output_map.dot(rotation_output_map);
}

template <class T>
void IntermediateTerm(const T *camera_parameters, const T *position,
                      T *residual, const T &u, const T &v) {
  using Tvector3d = Eigen::Matrix<T, 3, 1>;
  Tvector3d pij = UndistortionRay(camera_parameters + 6, u, v);

  Eigen::Map<const Tvector3d> translation(camera_parameters + 3);
  Eigen::Map<const Tvector3d> pj(position);
  T lambda = MinimumDepth(camera_parameters, position, u, v);
  Tvector3d pij_plus_lambda_multiple_translation = pij - lambda * translation;
  Eigen::Map<const Tvector3d> rotation_map(camera_parameters);
  Tvector3d rotation_inverse = rotation_map * T(-1.0);
  ceres::AngleAxisRotatePoint(rotation_inverse.data(), pij_plus_lambda_multiple_translation.data(), residual);

  Eigen::Map< Tvector3d> residual_map(residual);
  residual_map = T(0.5) * (residual_map  + lambda * pj);
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
    Eigen::Map<const Tvector3d> translation(camera_parameters + 3);
    T lambda_k = MinimumDepth(condition_camera_parameters,
                              condition_landmark_parameters, T(u_), T(v_));
    Tvector3d pij_minus_lambda_multiple_translation = pij - lambda_k * translation;
    Eigen::Map<const Tvector3d> rotation_map(camera_parameters);
    Tvector3d rotation_inverse = rotation_map * T(-1.0);
    ceres::AngleAxisRotatePoint(rotation_inverse.data(), pij_minus_lambda_multiple_translation.data(), residual);

    Tvector3d gij_k;
    IntermediateTerm(condition_camera_parameters, condition_landmark_parameters, gij_k.data(), T(u_), T(v_));
    
    Eigen::Map<Tvector3d> residual_map(residual);
    residual_map = residual_map - gij_k;
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
struct RayCostFunction {
  double u_, v_;
  RayCostFunction(double u, double v) : u_(u) , v_(v) {}

  template<class T>
  bool operator()(const T* camera_parameter, const T* landmark_parameter, T* residual) const {
    using Tvector3d = Eigen::Matrix<T, 3, 1>;
    Tvector3d pij = UndistortionRay(camera_parameter + 6, T(u_), T(v_));
    T lambda = MinimumDepth(camera_parameter, landmark_parameter, T(u_), T(v_));
    ceres::AngleAxisRotatePoint(camera_parameter, landmark_parameter, residual);

    Eigen::Map<const Tvector3d> translation(camera_parameter + 3);
    Eigen::Map<Tvector3d> residual_map(residual);
    residual_map = pij - lambda * (residual_map + translation);
    return true;
  }
};
#endif // BAL_COST_FUNCTION_AUTO_H_
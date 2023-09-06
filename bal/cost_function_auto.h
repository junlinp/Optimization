#ifndef BAL_COST_FUNCTION_AUTO_H_
#define BAL_COST_FUNCTION_AUTO_H_
#include "ceres/autodiff_cost_function.h"
#include "ceres/cost_function.h"
#include "ceres/rotation.h"

struct ProjectFunction {
  ProjectFunction(double u, double v) : u_(u), v_(v) {}

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

template <class T>
Eigen::Matrix<T, 3, 1> UndistortionRay(const T *focal_and_distortion, T u,
                                       T v) {
  const T &focal = focal_and_distortion[0];
  const T &k1 = focal_and_distortion[1];
  const T &k2 = focal_and_distortion[2];
  T l2_norm_uv = u * u + v * v;
  return Eigen::Matrix<T, 3, 1>(u, v,
                                focal + focal * k1 * l2_norm_uv +
                                    focal * k2 * l2_norm_uv * l2_norm_uv);
}

// Minimum lambda for |e_ij|^2 = min |p_ij - lambda * (R_i * l_j + t_i)|^2
// we got lambda = (p_ij).dot(R_i * l_j + t_i) / (R_i * l_j + t_i)^2
template <class T>
T MinimumDepth(const T *camera_parameters, const T *landmark_position, T u,
               T v) {
  using Tvector3d = Eigen::Matrix<T, 3, 1>;
  Tvector3d pij = UndistortionRay(camera_parameters + 6, u, v);
  Eigen::Map<const Tvector3d> ti(camera_parameters + 3);
  Eigen::Map<const Tvector3d> lj(landmark_position);
  T rotation_output[3];
  ceres::AngleAxisRotatePoint(camera_parameters, lj.data(), rotation_output);
  for (int i = 0; i < 3; i++) {
    rotation_output[i] += ti(i);
  }
  Eigen::Map<Tvector3d> rotation_output_map(rotation_output);
  return pij.dot(rotation_output_map) /
         rotation_output_map.dot(rotation_output_map);
}

template <class T>
void CameraTerm(const T *camera_parameters, const T &lambda,
                const T &u, const T &v, T *output) {
  using Tvector3d = Eigen::Matrix<T, 3, 1>;
  Tvector3d pij = UndistortionRay(camera_parameters + 6, u, v);
  T rotation_inverse[3] = {-camera_parameters[0], -camera_parameters[1],
                           -camera_parameters[2]};
  const T *translation = camera_parameters + 3;
  T t[3] = {pij[0] - lambda * translation[0], pij[1] - lambda * translation[1],
            pij[2] - lambda * translation[2]};
  ceres::AngleAxisRotatePoint(rotation_inverse, t, output);
}

template <class T>
void PointTerm(const T *point_position, const T &lambda, T *output) {
  output[0] = lambda * point_position[0];
  output[1] = lambda * point_position[1];
  output[2] = lambda * point_position[2];
}

inline void IntermediateTerm(const double *camera_parameters, const double *position,
                      double u, double v, double *residual) {

  double lambda = MinimumDepth(camera_parameters, position, u, v);

  double camera_term[3];
  double point_position_term[3];

  CameraTerm(camera_parameters, lambda, u, v, camera_term);
  PointTerm( position, lambda, point_position_term);
  residual[0] = 0.5 * (camera_term[0] + point_position_term[0]);
  residual[1] = 0.5 * (camera_term[1] + point_position_term[1]);
  residual[2] = 0.5 * (camera_term[2] + point_position_term[2]);
}

class CameraSurrogateCostFunction {
private:
  
  // we don't take the ownership.
  double* condition_camera_parameters_;
  double* condition_landmark_parameters_;
  double u_, v_;
public:
  CameraSurrogateCostFunction(double *condition_camera_parameters,
                              double *condition_landmark_parameters, double u,
                              double v)
      : condition_camera_parameters_(condition_camera_parameters),
        condition_landmark_parameters_(condition_landmark_parameters), u_(u),
        v_(v) {}

  template <class T>
  bool operator()(const T *camera_parameters, T *residual) const {
    double condition_lambda = MinimumDepth(condition_camera_parameters_,
                              condition_landmark_parameters_, u_, v_);
    T camera_term[3];
    CameraTerm(camera_parameters, T(condition_lambda), T(u_), T(v_), camera_term);
    double g[3];
    IntermediateTerm(condition_camera_parameters_,
                     condition_landmark_parameters_, u_, v_, g);

    for (int i = 0; i < 3; i++) {
      residual[i] = T(std::sqrt(2.0)) * (camera_term[i] - g[i]);
    }
    return true;
  }
};

class LandmarkSurrogatecostFunction {
private:
  // we don't take the ownership.
  double *condition_camera_parameters_;
  double *condition_landmark_parameters_;
  double u_, v_;

public:
  LandmarkSurrogatecostFunction(double *condition_camera_parameters,
                                double *condition_landmark_parameters, double u,
                                double v)
      : condition_camera_parameters_(condition_camera_parameters),
        condition_landmark_parameters_(condition_landmark_parameters), u_(u),
        v_(v) {}

  template <class T>
  bool operator()(const T *landmark_position, T *residual) const {
    double condition_lambda = MinimumDepth(
        condition_camera_parameters_, condition_landmark_parameters_, u_, v_);
    double g[3];
    IntermediateTerm(condition_camera_parameters_,
                     condition_landmark_parameters_, u_, v_, g);

    T point_term[3];
    PointTerm(landmark_position, T(condition_lambda), point_term);

    for (int i = 0; i < 3; i++) {
      residual[i] = T(std::sqrt(2.0)) * (point_term[i] - g[i]);
    }
    return true;
  }
};
struct RayCostFunction {
  double u_, v_;
  RayCostFunction(double u, double v) : u_(u), v_(v) {}

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

template<int DIM>
struct WeightVectorDiff {
  private:
  double* condition_parameters_;
  double weight_;
  public:
    WeightVectorDiff(double *condition_parameters, double weight)
        : condition_parameters_(condition_parameters), weight_(weight) {}

  template<class T>
  bool operator()(const T* parameter, T* residual) const {
     for(int i = 0; i < DIM; i++) {
       residual[i] =
           T(std::sqrt((weight_))) * (parameter[i] - T(condition_parameters_[i]));
     }
     return true;
  }

};
#endif // BAL_COST_FUNCTION_AUTO_H_
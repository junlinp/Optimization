#ifndef BAL_COST_FUNCTION_AUTO_H_
#define BAL_COST_FUNCTION_AUTO_H_
#include "ceres/autodiff_cost_function.h"
#include "ceres/cost_function.h"
#include "ceres/rotation.h"

struct ProjectFunction {
  ProjectFunction(double f, double k1, double k2, double u, double v)
      : focal_(f), k1_(k1), k2_(k2), u_(u), v_(v) {}
  static constexpr int CAMERA_PARAMETER_SIZE = 6;
  static const int POINT_PARAMETER_SIZE = 3;
  template <class T>
  bool operator()(const T *camera_param, const T *point, T *residual) const {
    T output_point[3];
    ceres::AngleAxisRotatePoint(camera_param, point, output_point);
    output_point[0] += camera_param[3];
    output_point[1] += camera_param[4];
    output_point[2] += camera_param[5];

    output_point[0] /= -output_point[2];
    output_point[1] /= -output_point[2];
    T focal = T(focal_);
    T K1 = T(k1_);
    T K2 = T(k2_);

    T p_norm_2 =
        output_point[0] * output_point[0] + output_point[1] * output_point[1];
    T distorsion = T(1.0) + K1 * p_norm_2 + K2 * p_norm_2 * p_norm_2;
    residual[0] = T(u_) - focal * distorsion * output_point[0];
    residual[1] = T(v_) - focal * distorsion * output_point[1];

    return true;
  }

  template <typename... ARGS>
  static ceres::CostFunction *CreateCostFunction(ARGS &&...args) {
    return new ceres::AutoDiffCostFunction<ProjectFunction, 2, CAMERA_PARAMETER_SIZE, POINT_PARAMETER_SIZE>(
        new ProjectFunction(std::forward<ARGS>(args)...));
  }

  double focal_, k1_, k2_;
  double u_, v_;
};
#endif // BAL_COST_FUNCTION_AUTO_H_
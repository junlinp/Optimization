#include "cost_function.h"

#include "ceres/rotation.h"

bool ProjectTransformCostFunction::Evaluate(const double *const *parameters,
                                            double *residual,
                                            double **jacobians) const {
  const double *point = parameters[0];
  residual[0] = point[0] / point[2];
  residual[1] = point[1] / point[2];
  if (jacobians != nullptr) {

    double *jacobian_u = jacobians[0];
    double *jacobian_v = jacobians[1];

    jacobian_u[0] = 1.0 / point[2];
    jacobian_u[1] = 0.0;
    jacobian_u[2] = -point[0] / point[2] / point[2];

    jacobian_v[0] = 0.0;
    jacobian_v[1] = 1.0 / point[2];
    jacobian_v[2] = -point[1] / point[2] / point[2];
  }
  return true;
}

bool RigidTransformCostFunction::Evaluate(const double *const *parameters,
                                          double *residuals,
                                          double **jacobians) const {
  double const *quaternion = parameters[0];
  const double *transform = parameters[1];
  const double *point = parameters[2];

  ceres::QuaternionRotatePoint(quaternion, point, residuals);
  for (int i = 0; i < 3; i++) {
    residuals[i] += transform[i];
  }
  // TODO(junlinp) : jacobians
  if (jacobians != nullptr) {
    double *jacobian_quaternion = jacobians[0];
    double *jacobian_transform = jacobians[1];
    double *jacobian_point = jacobians[2];

    if (jacobian_quaternion) {
      const double *v = quaternion + 1;
      const double w = quaternion[0];
      const double *a = point;
      // dw
      jacobian_quaternion[0 * 4 + 0] =
          2 * (w * a[0] - v[2] * a[1] + v[1] * a[2]);
      jacobian_quaternion[1 * 4 + 0] =
          2 * (v[3] * a[0] + w * a[1] - v[0] * a[2]);
      jacobian_quaternion[2 * 4 + 0] =
          2 * (-v[1] * a[0] + v[0] * a[1] + w * a[2]);
      double v_dot_a = v[0] * a[0] + v[1] + a[1] + v[2] + a[2];

      // dv
      jacobian_quaternion[0 * 4 + 1] = 2.0 * v_dot_a;
      jacobian_quaternion[0 * 4 + 2] = 2.0 * (v[0] * a[1] - a[0] * v[1] + a[2]);
      jacobian_quaternion[0 * 4 + 3] = 2.0 * (a[2] * v[0] - a[1] - a[0] * v[2]);

      jacobian_quaternion[1 * 4 + 1] = 2.0 * (v[1] * a[0] - a[1] * v[0] - a[2]);
      jacobian_quaternion[1 * 4 + 2] = 2.0 * v_dot_a;
      jacobian_quaternion[1 * 4 + 3] = 2.0 * (a[0] + a[2] * v[1] - a[1] * v[2]);

      jacobian_quaternion[2 * 4 + 1] = 2.0 * (a[1] + v[2] * a[0] - a[2] * v[0]);
      jacobian_quaternion[2 * 4 + 2] = 2.0 * (a[1] * v[2] - a[0] - a[2] * v[1]);
      jacobian_quaternion[2 * 4 + 3] = 2.0 * v_dot_a;
    }

    if (jacobian_transform) {
      // The jacobian matrix is identity for tranform
      std::fill_n(jacobian_transform, 3 * 3, 0.0);
      jacobian_transform[0] = 1.0;
      jacobian_transform[4] = 1.0;
      jacobian_transform[8] = 1.0;
    }

    if (jacobian_point) {
      // quaternion to rotation
      ceres::QuaternionToRotation(quaternion,
                                  ceres::RowMajorAdapter3x3(jacobian_point));
    }
  }
  return true;
}

bool RigidProjectTransformCostFunction::Evaluate(
    const double *const *parameters, double *residual,
    double **jacobians) const {
  double rigided_point[3] = {0.0};
  RigidTransformCostFunction rigid_transform;

  double rigid_quaternion_jacobians[3 * 4] = {0.0};
  double rigid_transform_jacobians[3 * 3] = {0.0};
  double rigid_point_jacobians[3 * 3] = {0.0};
  double *rigid_jacobians[3] = {rigid_quaternion_jacobians,
                                rigid_transform_jacobians,
                                rigid_point_jacobians};

  rigid_transform.Evaluate(parameters, rigided_point, rigid_jacobians);

  ProjectTransformCostFunction project_transform;
  double project_jacobian[2 * 3] = {0.0};
  const double *t_rigided_point = static_cast<const double *>(rigided_point);
  double *project_jacobians[1] = {project_jacobian};
  project_transform.Evaluate(&t_rigided_point, residual, project_jacobians);

  // TODO: compute jacobian
  if (jacobians) {

    double *jacobian_quaternion = jacobians[0];
    double *jacobian_transform = jacobians[1];
    double *jacobian_point = jacobians[2];

    if (jacobian_quaternion) {
      for (int row = 0; row < 2; row++) {
        for (int col = 0; col < 4; col++) {
          double c = 0.0;
          for (int k = 0; k < 3; k++) {
            c += project_jacobian[row * 3 + k] *
                 rigid_quaternion_jacobians[k * 4 + col];
          }
          jacobian_quaternion[row * 4 + col] = c;
        }
      }
    }

    if (jacobian_transform) {
      std::copy(project_jacobian, project_jacobian + 2 * 3, jacobian_transform);
    }

    if (jacobian_point) {

      for (int row = 0; row < 2; row++) {
        for (int col = 0; col < 3; col++) {
          double c = 0.0;
          for (int k = 0; k < 3; k++) {
            c += project_jacobian[row * 3 + k] *
                 rigid_point_jacobians[k * 3 + col];
          }
          jacobian_point[row * 4 + col] = c;
        }
      }
    }
  }
  return true;
}
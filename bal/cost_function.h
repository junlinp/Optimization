#ifndef BAL_COST_FUNCTION_H_
#define BAL_COST_FUNCTION_H_
#include "ceres/sized_cost_function.h"

#include "ceres/sized_cost_function.h"

class ProjectTransformCostFunction : public ceres::SizedCostFunction<2, 3> {
public:
  bool Evaluate(double const *const *parameters, double *residual,
                double **jacobians) const override;
  virtual ~ProjectTransformCostFunction() = default;
};

class RigidTransformCostFunction : public ceres::SizedCostFunction<3, 4, 3, 3> {
public:
  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override;

  virtual ~RigidTransformCostFunction() = default;
};

class RigidProjectTransformCostFunction : public ceres::SizedCostFunction<2, 4, 3, 3> {
  public:
  bool Evaluate(double const * const * parameters, double * residual, double** jacobians) const override;

  virtual ~RigidProjectTransformCostFunction() =default;
};
#endif // BAL_COST_FUNCTION_H_
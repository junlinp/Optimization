#ifndef RGB_RGD_H_
#define RGB_RGD_H_
#include "manifold.h"
#include "so3_cost_function_interface.h"

bool rgd(const SO3CostFunctionInterface &cost_function,
         std::vector<SO3Manifold::Vector> *x_init);

template <int Manifold, int residual_dimension> class CostFunctionFunctor {
public:
  using ResidualVector = Eigen::Matrix<double, residual_dimension, 1>;

  using JacobianMatrix =
      Eigen::Matrix<double, residual_dimension, Manifold::AmbientSpaceSize>;

  bool Evaluate(const Manifold *parameters, ResidualVector *residuals,
                JacobianMatrix *) const;
};

template <class CostFunctionFunctor, class Manifold>
bool LeastQuaresRiemannGredientDescent(
    CostFunctionFunctor &&costfunction_functor,
    Manifold &&manifold_init_value) {
  typename CostFunctionFunctor::ResidualVector residuals;
  typename CostFunctionFunctor::JacobianMatrix jacobians;
  costfunction_functor.Evaluate(&manifold_init_value, &residuals, &jacobians);
  
  double least_quares_error = 0.5 * residuals.squaredNorm();
  Eigen::VectorXd general_gradient = residuals.transpose() * jacobians;

  typename Manifold::TangentSpaceVector tangent_gradient =
      Manifold::Project(manifold_init_value, general_gradient);

  // linear search
  // return Step
}

#endif //  RGB_RGD_H_
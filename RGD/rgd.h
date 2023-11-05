#ifndef RGB_RGD_H_
#define RGB_RGD_H_
#include "manifold.h"
#include "so3_cost_function_interface.h"
#include <type_traits>

bool rgd(const SO3CostFunctionInterface &cost_function,
         std::vector<SO3Manifold::Vector> *x_init);

// template <int Manifold, int residual_dimension> 
// class CostFunctionFunctor {
// public:
//   using ResidualVector = Eigen::Matrix<double, residual_dimension, 1>;

//   using JacobianMatrix =
//       Eigen::Matrix<double, residual_dimension, Manifold::AmbientSpaceSize>;

//   bool Evaluate(const Manifold& parameters, ResidualVector *residuals,
//                 JacobianMatrix *) const;
// };

template <class CostFunctionFunctor, class Manifold>
bool LeastQuaresRiemannGredientDescentLinearSearch(
    CostFunctionFunctor &&costfunction_functor,
    Manifold &&manifold_init_value, typename std::remove_reference_t<Manifold>::TangentSpaceVector& step) {
      using OriginManifold = std::remove_reference_t<std::remove_cv_t<Manifold>>;

  typename std::remove_reference_t<CostFunctionFunctor>::ResidualVector residuals;
  typename std::remove_reference_t<CostFunctionFunctor>::JacobianMatrix jacobians;
  costfunction_functor.Evaluate(manifold_init_value, &residuals, &jacobians);
  
  double least_quares_error = 0.5 * residuals.squaredNorm();
  Eigen::VectorXd general_gradient = jacobians.transpose() * residuals;

  typename OriginManifold::TangentSpaceVector tangent_gradient =
      OriginManifold::Project(manifold_init_value, general_gradient);

  int max_iteration = 1024;
  int iteration = 0;
  double alpha = 1.0;
  double beta = 0.8;
  double eta = 1e-4;
  
  double gradient_norm =  tangent_gradient.squaredNorm();
  while(iteration++ < max_iteration) {
    OriginManifold next_step = OriginManifold::Retraction(manifold_init_value, alpha * tangent_gradient);
    costfunction_functor.Evaluate(next_step, &residuals, nullptr);
    double next_step_least_quares_error = 0.5 * residuals.squaredNorm();

    if ((least_quares_error - next_step_least_quares_error) < eta * alpha * gradient_norm) {
      step = alpha * tangent_gradient;
      return true;
    } else {
      alpha *= beta;
    }
  }
  return false;
}

#endif //  RGB_RGD_H_
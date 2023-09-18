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

    }

#endif //  RGB_RGD_H_
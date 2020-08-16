#ifndef OPTIMIZATION_BFGS_H_
#define OPTIMIZATION_BFGS_H_
#include "JET.h"
#include "linear_search.h"

template <int parameter_num>
void UpdateHessianInverse(
    Eigen::Matrix<double, parameter_num, parameter_num>& hessian_inverse,
    Eigen::Matrix<double, parameter_num, 1>& x_diff,
    Eigen::Matrix<double, parameter_num, 1>& gradient_diff) {
  double r = 1.0 / (x_diff.dot(gradient_diff) + 1e-11);
  Eigen::Matrix<double, parameter_num, parameter_num> identity =
      Eigen::Matrix<double, parameter_num, parameter_num>::Identity();
  hessian_inverse = (identity - r * x_diff * gradient_diff.transpose()) *
                        hessian_inverse *
                        (identity - r * gradient_diff * x_diff.transpose()) +
                    r * x_diff * x_diff.transpose();
}
template <class Functor, int residual_num, int parameter_num>
bool BFGS(AutoDiffFunction<Functor, residual_num, parameter_num>& functor,
          Eigen::Matrix<double, parameter_num, 1>& x0) {
  Eigen::Matrix<double, parameter_num, parameter_num> hessian_inverse =
      Eigen::Matrix<double, parameter_num, parameter_num>::Identity();
const double EPS = 1e-9;
  int max_num_iterator = 500;
  for (int iterator = 0; iterator < max_num_iterator; iterator++) {
      auto error = functor(x0);
      if (error.norm() < EPS) {
          return true;
      }
    auto gradient = functor.Jacobian(x0);
    Eigen::Matrix<double, parameter_num, 1> direct = -hessian_inverse * gradient;
    double t = BackTracing(functor, x0, direct);
    x0 = x0 + t * direct;
    Eigen::Matrix<double, parameter_num, 1> s = t * direct;
    auto new_gradient = functor.Jacobian(x0);
    Eigen::Matrix<double, parameter_num, 1> y = new_gradient - gradient;
    UpdateHessianInverse(hessian_inverse, s, y);
  }
  return true;
}
#endif  // OPTIMIZATION_BFGS_H_
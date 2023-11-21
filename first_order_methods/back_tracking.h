#ifndef FIRST_ORDER_METHODS_H_
#define FIRST_ORDER_METHODS_H_
#include "memory"
#include "Eigen/Dense"

template <class T>
double BackTracing(std::shared_ptr<T> cost_function, const Eigen::VectorXd &x,
                   const Eigen::VectorXd &directions) {
  double tau = 0.5;
  double r = 1e-4;
  double alpha = 1.0;
  double direction_norm = directions.norm();
  
  double fval_x = cost_function->fval(x);
  double it = 0;
  while ((fval_x - cost_function->fval(x - alpha * directions) <
         r * alpha * direction_norm) && it++ < 1024) {
    alpha *= tau;
  }
  return alpha;
}
#endif // FIRST_ORDER_METHODS_H_
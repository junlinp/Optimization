#ifndef OPTIMIZATION_LINEAR_SEARCH_H_
#define OPTIMIZATION_LINEAR_SEARCH_H_
#include "Eigen/Dense"

template<class Functor, int parameter_num>
double BackTracing(Functor& function_, Eigen::Matrix<double, parameter_num, 1>& x0, Eigen::Matrix<double, parameter_num, 1>& search_direct) {
   double t = 1.0; 
   double alpha = 0.2, beta = 0.8;
   function_(x0 + t * search_direct);
   while (function_(x0 + t * search_direct)(0) > function_(x0)(0) + alpha * t * function_.Jacobian(x0).dot(search_direct)) {
       t = beta * t;
       if (t < std::numeric_limits<double>::epsilon()) {
           return t;
       }
   }
   return t;
}
#endif  // OPTIMIZATION_LINEAR_SEARCH_H_
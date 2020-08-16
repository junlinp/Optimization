#ifndef OPTIMIZATION_BFGS_H_
#define OPTIMIZATION_BFGS_H_
#include "JET.h"


template<class Functor, int residual_num, int parameter_num>
bool BFGS(AutoDiffFunction<Functor, residual_num, parameter_num>& functor, Eigen::Matrix<double, parameter_num, 1>& x0) {

    return false;
}
#endif  // OPTIMIZATION_BFGS_H_
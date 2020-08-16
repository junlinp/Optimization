#ifndef OPTIMIZATION_GRADIENT_CHECKER_H_
#define OPTIMIZATION_GRADIENT_CHECKER_H_
#include "JET.h"
template<class Functor, int residual_num, int parameter_num>
bool GradientChecker(AutoDiffFunction<Functor, residual_num, parameter_num> functor, Eigen::Matrix<double, parameter_num, 1>& x0) {
    const double EPS = 1e-6;
    for (int residual_index = 0; residual_index < residual_num; residual_index++) {
        for (int parameter_index = 0; parameter_index < parameter_num; parameter_index++) {
            auto gradient = functor.Gradient(x0);
            auto auto_gradient = gradient(residual_index, parameter_index);

            auto x_plus = x0;
            x_plus(parameter_index) += EPS;
            auto x_sub = x0;
            x_sub(parameter_index) -= EPS;

            auto error = functor(x_plus) - functor(x_sub);
            auto computed_gradient = error(residual_index) / 2.0 / EPS;

            if (auto_gradient - computed_gradient > 1e-9) {
                std::cerr << "Gradient error " << std::endl;
                return false;
            }
        }
    }
    return true;
}

#endif // OPTIMIZATION_GRADIENT_CHECKER_H_
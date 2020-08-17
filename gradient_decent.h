#ifndef OPTIMIZATION_GRADIENT_DECENT_H_
#define OPTIMIZATION_GRADIENT_DECENT_H_
#include "JET.h"
#include "linear_search.h"
#include "glog/logging.h"

template<class Functor, int residual_num, int parameter_num>
bool GradientDecent(AutoDiffFunction<Functor, residual_num, parameter_num>& functor,
Eigen::Matrix<double, parameter_num, 1>& x0) {
    size_t max_iterator = 50000;
    double EPS = 1e-11;
    for(size_t i = 0; i < max_iterator; i++) {

        auto residual = functor(x0);
        //std::cout << "residual : " << residual << std::endl;
        Eigen::Matrix<double, parameter_num, 1> search_direct = -functor.Jacobian(x0);
        if (search_direct.norm() <EPS) {
            DLOG(INFO) << "residual : " << residual.transpose() << std::endl;
            return true;
        }
        //std::cout << " nagative gradient : " << search_direct.transpose() << std::endl;
        double step_size = BackTracing(functor, x0, search_direct);
        //std::cout << "step_size : " << step_size << std::endl;
        x0 = x0 + step_size * search_direct;
        DLOG(INFO) << "Move : " << step_size * search_direct.transpose() << std::endl; 
        DLOG(INFO) << "x : " << x0.transpose() << std::endl;
    }
    
    return true;
}
#endif // OPTIMIZATION_GRADIENT_DECNET_H_
#ifndef RGD_GRADIENT_CHECKER_H_
#define RGD_GRADIENT_CHECKER_H_
#include "rgd_cost_function_interface.h"
#include <memory>
#include <iostream>
class GradientChecker {
    public:
    template<class Manifold>
    static void Check(std::shared_ptr<RGDFirstOrderInterface> cost_function) {
        Eigen::VectorXd identity = Manifold::IdentityElement();
        Eigen::VectorXd jacobians = cost_function->Jacobian(identity);
        Eigen::VectorXd TxU = Manifold::Project(
            identity, jacobians);

        double t = 1e-8;
        double fval = cost_function->Evaluate(identity);
        while(t < 10) {

            double next_fval = cost_function->Evaluate(cost_function->Move(identity, t * TxU));
            double delta = next_fval - fval;
            // log(delta) ~= 2.0 * log(t) + const
            std::cout <<  std::log(t)
                      << "," << std::log(delta) << std::endl;
            t *= 1.5;
        }
    }
};
#endif  // RGD_GRADIENT_CHECKER_H_
#ifndef RGD_GRADIENT_CHECKER_H_
#define RGD_GRADIENT_CHECKER_H_
#include "rgd_cost_function_interface.h"
#include <memory>
#include <iostream>
class GradientChecker {
    public:
    template<class Manifold>
    static void Check(std::shared_ptr<RGDFirstOrderInterface> cost_function) {
        Eigen::VectorXd x = Manifold::RandomElement();
        Eigen::VectorXd jacobians = cost_function->Jacobian(x);
        Eigen::VectorXd gradient = Manifold::Project(
            x, jacobians);

        assert(Manifold::IsTangentSpaceVector(gradient));

        double t = 1e-8;
        double fval = cost_function->Evaluate(x);
        Eigen::Vector3d random;
        random.setRandom();

        Eigen::VectorXd v = gradient;
        v.setRandom();
        v = Manifold::Project(x, v);
        v.normalized();
        assert(Manifold::IsTangentSpaceVector(v));
        double last_delta = 0.0;
        double last_t = 0.0;

        double Df_v = gradient.dot(v);
        while(t < 10) {
            double next_fval = cost_function->Evaluate(cost_function->Move(x, t * v));
            double delta = std::abs(next_fval - fval - t * Df_v);
            // log(delta) ~= 2.0 * log(t) + const
            std::cout <<  std::log(t)
                      << "," << std::log(delta) << " slope : " << (std::log(delta) - last_delta) / (std::log(t) - last_t) << std::endl;
            last_delta = std::log(delta);
            last_t = std::log(t);
            t *= 1.1;
        }
    }
};
#endif  // RGD_GRADIENT_CHECKER_H_
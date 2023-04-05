#include "rgd.h"
#include "Eigen/Householder"
#include "iostream"
#include "so3_cost_function_interface.h"

bool rgd(const SO3CostFunctionInterface &cost_function,
         std::vector<SO3Manifold::Vector> *x_init) {
  size_t max_iteration = 129;
  size_t iteration = 0;
  std::cout << "Initial error : " << cost_function.Evaluate(*x_init) << std::endl;
  while (iteration++ < max_iteration) {
    auto jacobians = cost_function.Jacobian(*x_init);
    const double step = 0.01;

    for (size_t i = 0; i < x_init->size(); i++) {
      SO3Manifold::TangentVector TxU = SO3Manifold::Project((*x_init)[i], jacobians[i]);

      if (!SO3Manifold::CheckTangentVector((*x_init)[i], TxU)) {
        std::cout << "CheckTangent False" << std::endl;
        return false;
      }

      std::cout << "TxU : " << TxU << std::endl;
      SO3Manifold::TangentVector sk = -step * TxU;
      (*x_init)[i] = SO3Manifold::Retraction((*x_init)[i], sk);

    }
    std::cout << "error : " << cost_function.Evaluate(*x_init) << std::endl;
  }
  return true;
}
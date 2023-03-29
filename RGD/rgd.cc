#include "rgd.h"
#include "Eigen/Householder"
#include "iostream"
Eigen::Matrix3d SkewPart(const Eigen::Matrix3d& X) {
    return 0.5 * (X.transpose() - X);
}

Eigen::Matrix3d Project(const Eigen::Matrix3d& X, const Eigen::Matrix3d& U) {
    std::cout << "Project X : " << X << std::endl;
    std::cout << "Project U : " << U << std::endl;
    std::cout << "X.transpose() * U : " << X.transpose() * U << std::endl;
    std::cout << "Skew Part : " << SkewPart(X.transpose() * U) << std::endl;
    return X * SkewPart(X.transpose() * U);
}

Eigen::Matrix3d Retraction(const Eigen::Matrix3d& X, const Eigen::Matrix3d& V) {
    return (X + V).householderQr().householderQ();
}

bool rgd(const SO3CostFunctionInterface &cost_function,
         std::vector<Eigen::Matrix3d> *x_init) {
  size_t max_iteration = 1024;
  size_t iteration = 0;
  while (iteration++ < max_iteration) {
    auto jacobian = cost_function.Jacobian(*x_init);
    const double step = 0.01;

    for (size_t i = 0; i < x_init->size(); i++) {
      Eigen::Matrix3d TxU = Project((*x_init)[i], jacobian[i]);
      std::cout << "TxU : " << TxU << std::endl;
      Eigen::Matrix3d sk = step * TxU;
      (*x_init)[i] = Retraction((*x_init)[i], sk);
    }
  }
  return true;
}
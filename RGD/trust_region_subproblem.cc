#include "RGD/trust_region_subproblem.h"
#include <limits>
#include <iostream>

bool TrustRegionSubProblem(const Eigen::MatrixXd &A, const Eigen::VectorXd &B,
                           double trust_region_radius,
                           Eigen::VectorXd *initial_solution) {
  Eigen::VectorXd p = *initial_solution;

  double lambda = 0;

  double gradient_error =
      ((A + lambda * Eigen::MatrixXd::Identity(p.size(), p.size())) * p + B)
          .norm();

  std::cout << "Init Gradient Error: " << gradient_error << std::endl;
  while (gradient_error > 1e-6) {
    auto solution = (A + lambda * Eigen::MatrixXd::Identity(p.size(), p.size())).llt();
    Eigen::MatrixXd L = solution.matrixL();
    std::cout << "Matrix L : " << L << std::endl;
    p = solution.solve(-B);
    Eigen::VectorXd q = L.triangularView<Eigen::Lower>().solve(p);
    lambda += (p.dot(p)) / (q.dot(q)) * (p.norm() - trust_region_radius) / trust_region_radius;

    gradient_error =
        ((A + lambda * Eigen::MatrixXd::Identity(p.size(), p.size())) * p + B)
            .norm();
  }
  std::cout << "Gradient Error: " << gradient_error << std::endl;
  std::cout << "lambda : " << lambda << std::endl;
  std::cout << "p : " << p.norm() << std::endl;

  *initial_solution = p;
  return true;
}

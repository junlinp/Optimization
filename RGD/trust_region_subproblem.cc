#include "RGD/trust_region_subproblem.h"
#include <limits>
#include <iostream>

#include "RGD/conjugate_gradient.h"

bool TrustRegionSubProblem(const Eigen::MatrixXd &H, const Eigen::VectorXd &g,
                           double trust_region_radius,
                           Eigen::VectorXd *initial_solution) {
  Eigen::VectorXd p = *initial_solution;

  double lambda = 0;

  Eigen::VectorXd p_b;

  ConjugateGradient(H, g, 1e-6, &p_b);

  if (p_b.norm() < trust_region_radius){
    *initial_solution = p_b;
    return true;
  } 

  Eigen::VectorXd p_u = g.dot(g) / g.dot(H * g) * g;

  if (p_u.norm() > trust_region_radius) {
    *initial_solution = trust_region_radius * p_u / p_u.norm();
    return true;
  }

  double a = (p_b - p_u).squaredNorm();
  double b = 2.0 * p_u.dot(p_b - p_u);
  double c = p_u.squaredNorm() - trust_region_radius;

  if (b * b - 4 * a * c < 0) {
    return false;
  }

  double tau = (b * b - std::sqrt(4 * a * c)) / 2 / a;
  if (tau < 0 || tau > 1.0) {
    return false;
  }
  *initial_solution = p_u + tau * (p_b - p_u);
  return true;
}

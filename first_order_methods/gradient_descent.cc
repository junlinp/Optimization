#include "gradient_descent.h"
#include "memory"
Eigen::VectorXd GradientDescnet(std::shared_ptr<FirstOracle> function, const Eigen::VectorXd& x0, double L) {

    Eigen::VectorXd y = x0;
    constexpr int max_iteration = 1024;
    int iteration = 0;

    while (iteration++ < max_iteration) {
        Eigen::VectorXd J = function->SubGradient(y);
        y = y - 1.0 / L * J;
    }

    return y;
}

Eigen::VectorXd  NesterovGradientDescnet(std::shared_ptr<FirstOracle> function, const Eigen::VectorXd& x0, double L) {
  Eigen::VectorXd y = x0;
  Eigen::VectorXd z = x0;
  constexpr int max_iteration = 1024;
  int iteration = 0;
  double mu = 1;
  while (iteration++ < max_iteration) {
    double mu_plus = (1 + std::sqrt(1 + 4 * mu * mu)) * 0.5;
    double beta = (mu - 1) / mu_plus;
    mu = mu_plus;
    Eigen::VectorXd J = function->SubGradient(y);
    Eigen::VectorXd z_plus = y - 1.0 / L * J;
    y = z_plus + beta * (z_plus - z);
    z = z_plus;
    }
    return y;
}

Eigen::VectorXd  NesterovGradientDescnet2(std::shared_ptr<FirstOracle> function, const Eigen::VectorXd& x0, double L) {

  Eigen::VectorXd x = x0;
  Eigen::VectorXd y = x0;
  Eigen::VectorXd z = x0;
  constexpr int max_iteration = 1024;
  int iteration = 0;
  while (iteration++ < max_iteration) {
    double alpha = (iteration + 2) * 2 / L;
    double tau = 2.0 / (iteration + 2);
    x = tau * z + (1-tau) * y;
    Eigen::VectorXd J = function->SubGradient(x);
    y = x - 1.0 / L * J;
    z = z - alpha * J;
  }
  return y;
}

#include "linear_programing.h"

#include "iostream"

void LPSolver(const Eigen::VectorXd& c, const Eigen::MatrixXd& A,
              const Eigen::VectorXd& b, Eigen::VectorXd& x) {
  size_t n = c.rows();
  size_t m = A.rows();

  x = Eigen::VectorXd::Ones(n);
  Eigen::VectorXd z = Eigen::VectorXd::Ones(n);
  Eigen::VectorXd y = Eigen::VectorXd::Ones(m);

  double mu = 0.0;
  Eigen::VectorXd e = Eigen::VectorXd::Ones(n);
  for (int iter = 0; iter < 16; iter++) {
    Eigen::MatrixXd Z = z.asDiagonal();
    Eigen::MatrixXd X = x.asDiagonal();
    Eigen::MatrixXd H(2 * n + m, 2 * n + m);
    H << A, Eigen::MatrixXd::Zero(m, m), Eigen::MatrixXd::Zero(m, n),
        Eigen::MatrixXd(n, n), A.transpose(), Eigen::MatrixXd::Identity(n, n),
        Z, Eigen::MatrixXd::Zero(n, m), X;
    std::cout << "H : " << H << std::endl;
    Eigen::VectorXd B(2 * n + m);
    B << b - A * x, c - z - A.transpose() * y, X * Z * e - mu * e;

    Eigen::VectorXd delta = H.inverse() * B;
    std::cout << "delta : " << delta << std::endl;
    Eigen::VectorXd delta_x = delta.block(0, 0, n, 1);
    Eigen::VectorXd delta_y = delta.block(n, 0, m, 1);
    Eigen::VectorXd delta_z = delta.block(n + m, 0, n, 1);

    double alpha = 1.0;
    for (int i = 0; i < n; i++) {
      double a = x(i) / delta_x(i);
      double b = z(i) / delta_z(i);

      if (0.0 < a && a < 1.0) {
        alpha = std::min(alpha, a);
      }

      if (0.0 < b && b < 1.0) {
        alpha = std::min(alpha, b);
      }
    }
    std::cout << "alpha : " << alpha << std::endl;
    //if (delta_x.dot(delta_z) < n * 1e-8) {
    //    return;
    //}
    x = x + alpha * delta_x;
    y = y + alpha * delta_y;
    z = z + alpha * delta_z;
  }
}

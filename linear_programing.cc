#include "linear_programing.h"

#include "iostream"

void LPSolver(const Eigen::VectorXd& c, const Eigen::MatrixXd& A,
              const Eigen::VectorXd& b, Eigen::VectorXd& x) {
  size_t n = c.rows();
  size_t m = A.rows();

  x = Eigen::VectorXd::Ones(n);
  Eigen::VectorXd z = Eigen::VectorXd::Ones(n);
  Eigen::VectorXd y = Eigen::VectorXd::Zero(m);
  double eps_feas = 1e-5;
  double eps = 1e-8;
  size_t max_iterator = 1024;
  double mu = 10.0;
  double alpha = 0.1;
  double beta = 0.5;
  Eigen::MatrixXd zero_m_m(m, m), zero_m_n(m, n), zero_n_n(n, n);
  for (int i = 0; i < m; i++) {
      for (int j = 0;j < m; j++) {
          zero_m_m(i, j) = 0.0;
      }
      for (int k = 0; k < n; k++) {
          zero_m_n(i, k) = 0.0;
      }
  }

  for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
          zero_n_n(i, j) = 0.0;
      }
  }

  Eigen::VectorXd e = Eigen::VectorXd::Ones(n);
  for (int iter = 0; iter < max_iterator; iter++) {
    double t = mu * m / x.dot(z);
    Eigen::MatrixXd Z = z.asDiagonal();
    Eigen::MatrixXd X = x.asDiagonal();
    Eigen::MatrixXd H(2 * n + m, 2 * n + m);
    H << A, zero_m_m, zero_m_n,
        zero_n_n, A.transpose(), Eigen::MatrixXd::Identity(n, n),
        Z, zero_m_n.transpose(), X;
    //std::cout << "H : " << H << std::endl;
    Eigen::VectorXd B(2 * n + m);
    B << b - A * x, c - z - A.transpose() * y, mu * e -X * Z * e ;

    Eigen::VectorXd delta = H.fullPivLu().solve(B);
    std::cout << "delta : " << delta << std::endl;
    Eigen::VectorXd delta_x = delta.block(0, 0, n, 1);
    Eigen::VectorXd delta_y = delta.block(n, 0, m, 1);
    Eigen::VectorXd delta_z = delta.block(n + m, 0, n, 1);

    double s_max = 1.0;
    for (int i = 0; i < n; i++) {
      if (delta_x(i) < 0) {
        s_max = std::min(s_max, x(i) / -delta_x(i));
      }

      if (delta_z(i) < 0) {
          s_max = std::min(s_max, z(i) / -delta_z(i));
      }

    }
    std::cout << "s_max : " << s_max << std::endl;
    
    double s = 0.95 * s_max;

    std::cout << "Step : " << s << std::endl;
    x = x + s * delta_x;
    y = y + s * delta_y;
    z = z + s * delta_z;
    std::cout << "Function : " << c.dot(x) << std::endl;
    mu *= 0.01;
    if ((A * x - b).norm() <= eps_feas && (c - z - A.transpose() * y).norm() <= eps_feas && (x.dot(z) < eps)) {
        std::cout << "Minimum Found" << std::endl;
        break;
    }
  }
}

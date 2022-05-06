#include "linear_programing.h"

#include <assert.h>

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
  double sigma = 0.9;
  Eigen::MatrixXd zero_m_m(m, m), zero_m_n(m, n), zero_n_n(n, n);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
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
  std::cout << "Iterator\t\tPrimal\t\tDual" << std::endl;
  for (int iter = 0; iter < max_iterator; iter++) {
    Eigen::MatrixXd Z = z.asDiagonal();
    Eigen::MatrixXd X = x.asDiagonal();
    Eigen::MatrixXd H(2 * n + m, 2 * n + m);
    H << A, zero_m_m, zero_m_n, zero_n_n, A.transpose(),
        Eigen::MatrixXd::Identity(n, n), Z, zero_m_n.transpose(), X;
    // std::cout << "H : " << H << std::endl;
    Eigen::VectorXd B(2 * n + m);
    double t = (1 - sigma / std::sqrt(n)) * x.dot(z) / n;
    B << b - A * x, c - z - A.transpose() * y, t * e - X * Z * e;

    Eigen::VectorXd delta = H.fullPivLu().solve(B);
    // std::cout << "delta : " << delta << std::endl;
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
    // std::cout << "s_max : " << s_max << std::endl;
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd;
    // svd.compute(X.inverse() * x.asDiagonal());
    // std::cout << "Largest EigenValue : " << svd.singularValues().maxCoeff()
    // << std::endl; Eigen::JacobiSVD<Eigen::MatrixXd> svd2;
    // svd2.compute(Z.inverse() * z.asDiagonal());
    // s_max = 1.0 / std::max(svd.singularValues().maxCoeff(),
    // svd2.singularValues().maxCoeff());
    double s = 0.95 * s_max;

    // std::cout << "Step : " << s << std::endl;
    x = x + s * delta_x;
    y = y + s * delta_y;
    z = z + s * delta_z;
    // std::cout << "Function : " << c.dot(x) << std::endl;
    mu *= 0.01;
    std::cout << iter << "\t\t" << (A * x - b).norm() << "\t\t"
              << (c - z - A.transpose() * y).norm() << std::endl;
    if ((A * x - b).norm() <= eps_feas &&
        (c - z - A.transpose() * y).norm() <= eps_feas && (x.dot(z) < eps)) {
      std::cout << "Minimum Found" << std::endl;
      break;
    }
  }
}

Eigen::VectorXd vec(const Eigen::MatrixXd& A) {
  int n = A.cols();
  int m = A.rows();
  Eigen::VectorXd res(m * n);
  for (int i = 0; i < n; i++) {
    res.block(m * i, 0, m, 1) = A.col(i);
  }
  return res;
}

Eigen::VectorXd svec(const Eigen::MatrixXd& A) {
  assert(A.rows() == A.cols());
  const int n = A.rows();
  const int vector_dimension = (n * (n + 1)) >> 1;
  Eigen::VectorXd res(vector_dimension);

  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      int vector_index = ((2 * n + 1 - i) * i) + j;
      res(vector_index) = A(i, j);
    }
  }
  return res;
}
/**
 *  tr(A * X) can be represent as the inter product of L(A) and svec(X)
 *
 */
Eigen::VectorXd L(const Eigen::MatrixXd& A) {
  assert(A.rows() == A.cols());
  const int n = A.rows();
  const int vector_dimension = (n * (n + 1)) >> 1;
  Eigen::VectorXd res(vector_dimension);

  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      int vector_index = ((2 * n + 1 - i) * i) + j;
      res(vector_index) = A(i, j) + A(j, i);
    }
  }
  return res;
}

Eigen::MatrixXd KroneckerProduct(const Eigen::MatrixXd& A,
                                 const Eigen::MatrixXd& B) {
  Eigen::MatrixXd res(A.rows() * B.rows(), A.cols() * B.cols());

  for (int i = 0; i < A.rows(); i++) {
    for (int j = 0; j < A.cols(); j++) {
      res.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
    }
  }
  return res;
}
Eigen::MatrixXd KroneckerSum(const Eigen::MatrixXd& A,
                             const Eigen::MatrixXd& B) {
  return KroneckerProduct(A, B) + KroneckerProduct(B, A);
}

Eigen::MatrixXd mat(const Eigen::VectorXd& vec) {
  size_t n = static_cast<size_t>(std::sqrtf(vec.rows()) + 0.5);
  Eigen::MatrixXd res(n, n);

  for (int i = 0; i < n; i++) {
    res.col(i) = vec.block(n * i, 0, n, 1);
  }
  return res;
}
double ComputeMinimumEigenValue(const Eigen::MatrixXd& Matrix) {
  Eigen::EigenSolver<Eigen::MatrixXd> eigen(Matrix);
  // std::cout << "singular : " << singular << std::endl;
  auto eigen_value = eigen.eigenvalues();
  return eigen_value(eigen_value.rows() - 1).real();
}

double ComputeLargestSingularValue(const Eigen::MatrixXd& Matrix) {
  Eigen::BDCSVD<Eigen::MatrixXd> svd;
  svd.compute(Matrix);
  auto singular = svd.singularValues();
  return singular(0);
}
void SymmetricSolver(const Eigen::MatrixXd C,
                     const std::vector<Eigen::MatrixXd>& A,
                     const Eigen::VectorXd& b, Eigen::MatrixXd& x) {
  // assume C is a square matrix
  assert(C.rows() == C.cols());
  const int n = C.rows();
  const int m = A.size();
  Eigen::MatrixXd A_(m, n * n);
  for (int i = 0; i < m; i++) {
    A_.row(i) = vec(A[i]);
  }
  double mu = 0.1;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  x = I;
  Eigen::MatrixXd z = I;
  Eigen::VectorXd y = Eigen::VectorXd::Zero(m);
  Eigen::MatrixXd zero_m_m(m, m), zero_m_n(m, n * n), zero_n_n(n * n, n * n);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      zero_m_m(i, j) = 0.0;
    }
    for (int k = 0; k < n * n; k++) {
      zero_m_n(i, k) = 0.0;
    }
  }

  for (int i = 0; i < n * n; i++) {
    for (int j = 0; j < n * n; j++) {
      zero_n_n(i, j) = 0.0;
    }
  }
  const size_t max_iterator = 16;
  for (size_t iterator = 0; iterator < max_iterator; iterator++) {
    Eigen::MatrixXd X = x;
    Eigen::MatrixXd Z = z;
    x = X;
    Eigen::MatrixXd Jacobian(m + 2 * n * n, m + 2 * n * n);
    Eigen::VectorXd g(m + 2 * n * n);
    Jacobian << A_, zero_m_m, zero_m_n, zero_n_n, A_.transpose(),
        Eigen::MatrixXd::Identity(n * n, n * n),
        KroneckerProduct(Z, I) + KroneckerProduct(I, Z), zero_m_n.transpose(),
        KroneckerProduct(X, I) + KroneckerProduct(I, X);

    g << b - A_ * vec(X), vec(C) - vec(Z) - A_.transpose() * y,
        vec(2 * mu * I - X * Z - Z * X);

    Eigen::VectorXd delta = Jacobian.fullPivLu().solve(g);
    Eigen::VectorXd delta_x = delta.block(0, 0, n * n, 1);
    Eigen::VectorXd delta_y = delta.block(n * n, 0, m, 1);
    Eigen::VectorXd delta_z = delta.block(n * n + m, 0, n * n, 1);

    double s_a = 1.0;
    while (ComputeMinimumEigenValue(x + s_a * mat(delta_x)) < 1e-6) {
      s_a *= 0.9;
    }
    double s_b = 1.0;
    while (ComputeMinimumEigenValue(z + s_b * mat(delta_z)) < 1e-6) {
      s_b *= 0.9;
    }
    std::cout << "Largest X : " << s_a << std::endl;
    std::cout << "Largest Z : " << s_b << std::endl;
    // std::cout << "x + s * mat(delta_x) : " << ComputeMinimumSingularValue(x +
    // s * mat(delta_x)) << std::endl;

    std::cout << s_a << std::endl;
    std::cout << s_b << std::endl;
    x = x + s_a * mat(delta_x);
    y = y + s_b * delta_y;
    z = z + s_b * mat(delta_z);
    Eigen::BDCSVD<Eigen::MatrixXd> svd;
    svd.compute(x);
    std::cout << "singular : " << svd.singularValues() << std::endl;
    std::cout << "X : " << x << std::endl;
    std::cout << iterator << "\t\t" << (A_ * vec(x) - b).norm() << "\t\t"
              << (vec(C) - vec(Z) - A_.transpose() * y).norm() << "\t\t"
              << (X * Z).norm() << std::endl;
    mu *= 0.0001;
  }
}




void LPSolver2(const Eigen::VectorXd& c, const Eigen::MatrixXd& A,
               const Eigen::VectorXd& b, Eigen::VectorXd& x) {
      FullNTStepIMP(c, A, b, x, OrthantSpace{});
}

void SymmetricSolver2(const Eigen::MatrixXd C,
                      const std::vector<Eigen::MatrixXd>& A,
                      const Eigen::VectorXd& b, Eigen::MatrixXd& x) {
  // Convert the SDP to LCP Problem

  //   min tr(C * X)
  //   s.t tr(A_i * X) = b_i for i = 0... m
  //       X >= 0  for X is a semidefinite cone

  // Using KKT Conditions
  // we have
  //
}


void RobustLPSolver(const Eigen::VectorXd& c,
                    const std::vector<Eigen::VectorXd>& A,
                    const std::vector<double>& b,
                    const std::vector<Eigen::VectorXd>& E,
                    const std::vector<double>& F, Eigen::VectorXd& x) {
 Eigen::MatrixXd A0(A.size(), A.at(0).rows());
 Eigen::VectorXd b0(b.size());
                    }



void SDPSolver(const Eigen::VectorXd& c, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::MatrixXd& X) {
  Eigen::VectorXd x(c.rows() * c.rows());
  FullNTStepIMP(c, A, b, x, SemiDefineSpace{});
  X = SemiDefineSpace::Mat(x);
}
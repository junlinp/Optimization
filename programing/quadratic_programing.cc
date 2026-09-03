#include "quadratic_programing.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

// Infeasible-start primal-dual interior-point method for convex QP:
//
//   min  0.5 x^T H x + g^T x
//   s.t. A x = b   (dual y)
//        C x <= d  (slack s >= 0, dual z >= 0)
//
// KKT stationarity residual : r_dual = H x + g + A^T y + C^T z
// Equality residual         : r_eq   = A x - b
// Inequality residual       : r_ineq = C x + s - d
// Complementarity residual  : r_comp = s .* z - t * e   (t: centering target)
//
// Each iteration solves the Newton system for [dx, dy, dz, ds]:
//
//   H dx + A^T dy + C^T dz         = -r_dual
//   A dx                           = -r_eq
//   C dx                    + ds   = -r_ineq
//              S dz + Z ds         = -r_comp
//
// then takes a fraction-to-boundary step keeping s, z > 0.
bool QPSolver(const Eigen::MatrixXd& H, const Eigen::VectorXd& g,
              const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
              const Eigen::MatrixXd& C, const Eigen::VectorXd& d,
              Eigen::VectorXd& x) {
  const Eigen::Index n = g.rows();
  const Eigen::Index m_eq = A.rows();
  const Eigen::Index m_ineq = C.rows();

  x = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd y = Eigen::VectorXd::Zero(m_eq);
  Eigen::VectorXd s = Eigen::VectorXd::Ones(m_ineq);
  Eigen::VectorXd z = Eigen::VectorXd::Ones(m_ineq);

  const double eps_feas = 1e-8;
  const double eps_gap = 1e-10;
  const double sigma = 0.1;
  const double tau = 0.95;
  const size_t max_iterator = 200;

  const Eigen::Index dim = n + m_eq + 2 * m_ineq;
  typedef Eigen::Triplet<double> T;

  for (size_t iter = 0; iter < max_iterator; iter++) {
    Eigen::VectorXd r_dual = H * x + g + A.transpose() * y + C.transpose() * z;
    Eigen::VectorXd r_eq = A * x - b;
    Eigen::VectorXd r_ineq = C * x + s - d;
    double mu = m_ineq > 0 ? s.dot(z) / static_cast<double>(m_ineq) : 0.0;

    if (r_dual.norm() <= eps_feas && r_eq.norm() <= eps_feas &&
        r_ineq.norm() <= eps_feas && mu <= eps_gap) {
      std::cout << "QPSolver: converged at iteration " << iter << std::endl;
      return true;
    }

    double t = sigma * mu;

    std::vector<T> coefficients;
    // Row block 0: dual stationarity, rows [0, n).
    for (Eigen::Index row = 0; row < n; row++) {
      for (Eigen::Index col = 0; col < n; col++) {
        if (H(row, col) != 0.0) coefficients.push_back(T(row, col, H(row, col)));
      }
      for (Eigen::Index k = 0; k < m_eq; k++) {
        if (A(k, row) != 0.0) coefficients.push_back(T(row, n + k, A(k, row)));
      }
      for (Eigen::Index j = 0; j < m_ineq; j++) {
        if (C(j, row) != 0.0)
          coefficients.push_back(T(row, n + m_eq + j, C(j, row)));
      }
    }
    // Row block 1: primal equality residual, rows [n, n + m_eq).
    for (Eigen::Index row = 0; row < m_eq; row++) {
      for (Eigen::Index col = 0; col < n; col++) {
        if (A(row, col) != 0.0)
          coefficients.push_back(T(n + row, col, A(row, col)));
      }
    }
    // Row block 2: primal inequality residual, rows [n+m_eq, n+m_eq+m_ineq).
    for (Eigen::Index row = 0; row < m_ineq; row++) {
      for (Eigen::Index col = 0; col < n; col++) {
        if (C(row, col) != 0.0)
          coefficients.push_back(T(n + m_eq + row, col, C(row, col)));
      }
      coefficients.push_back(
          T(n + m_eq + row, n + m_eq + m_ineq + row, 1.0));
    }
    // Row block 3: complementarity, rows [n+m_eq+m_ineq, dim).
    for (Eigen::Index row = 0; row < m_ineq; row++) {
      coefficients.push_back(
          T(n + m_eq + m_ineq + row, n + m_eq + row, s(row)));
      coefficients.push_back(
          T(n + m_eq + m_ineq + row, n + m_eq + m_ineq + row, z(row)));
    }

    Eigen::SparseMatrix<double> KKT(dim, dim);
    KKT.setFromTriplets(coefficients.begin(), coefficients.end());

    Eigen::VectorXd rhs(dim);
    rhs << -r_dual, -r_eq, -r_ineq,
        t * Eigen::VectorXd::Ones(m_ineq) - s.cwiseProduct(z);

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(KKT);
    Eigen::VectorXd delta;
    if (solver.info() == Eigen::Success) {
      delta = solver.solve(rhs);
    }
    if (solver.info() != Eigen::Success) {
      delta = Eigen::MatrixXd(KKT).fullPivLu().solve(rhs);
    }

    Eigen::VectorXd delta_x = delta.segment(0, n);
    Eigen::VectorXd delta_y = delta.segment(n, m_eq);
    Eigen::VectorXd delta_z = delta.segment(n + m_eq, m_ineq);
    Eigen::VectorXd delta_s = delta.segment(n + m_eq + m_ineq, m_ineq);

    double step = 1.0;
    for (Eigen::Index i = 0; i < m_ineq; i++) {
      if (delta_s(i) < 0) step = std::min(step, -s(i) / delta_s(i));
      if (delta_z(i) < 0) step = std::min(step, -z(i) / delta_z(i));
    }
    step *= tau;

    x += step * delta_x;
    y += step * delta_y;
    z += step * delta_z;
    s += step * delta_s;

    std::cout << "QPSolver iter " << iter << "\tdual " << r_dual.norm()
              << "\teq " << r_eq.norm() << "\tineq " << r_ineq.norm()
              << "\tgap " << mu << std::endl;
  }

  std::cout << "QPSolver: max iterations reached without convergence"
            << std::endl;
  return false;
}

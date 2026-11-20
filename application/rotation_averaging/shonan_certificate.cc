#include "application/rotation_averaging/shonan_certificate.h"

#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace shonan_internal {
namespace {

// Bound lambda_min of a fully stored real symmetric certificate. A successful
// factorization of C - sI proves s < lambda_min in exact arithmetic. Rayleigh
// quotients give upper bounds only. Reuse the symbolic ordering for every
// shift, and factor sparsely throughout. Cholesky fill depends on graph order.
bool sparseCertificateEigenpair(const Eigen::SparseMatrix<double>& certificate,
                                double target_shift, double* eigenvalue_lower_bound,
                                Eigen::VectorXd* direction) {
  const Eigen::Index n = certificate.rows();
  if (n == 0 || certificate.cols() != n || !std::isfinite(target_shift) ||
      target_shift <= 0.0) return false;

  double lower = std::numeric_limits<double>::infinity();
  double upper = lower;
  double norm = 0.0;
  Eigen::Index min_index = 0;
  for (Eigen::Index col = 0; col < n; ++col) {
    double diagonal = 0.0;
    long double radius = 0.0;
    for (Eigen::SparseMatrix<double>::InnerIterator it(certificate, col); it; ++it) {
      if (!std::isfinite(it.value())) return false;
      if (it.row() == col) diagonal = it.value();
      else radius += std::abs(it.value());
    }
    if (diagonal < upper) {
      upper = diagonal;
      min_index = col;
    }
    lower = std::min(lower, static_cast<double>(diagonal - radius));
    norm = std::max(norm, static_cast<double>(std::abs(diagonal) + radius));
  }
  if (!std::isfinite(norm)) return false;
  // Leave a roundoff margin on floating-point Cholesky sign decisions.
  const double guard = (32.0 * std::numeric_limits<double>::epsilon()) * norm;
  const double accuracy = std::max(0.0625 * target_shift, 8.0 * guard);
  direction->setZero(n);
  (*direction)(min_index) = 1.0;
  double rayleigh = upper;

  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> factor;
  factor.analyzePattern(certificate);
  std::mt19937_64 generator(0x53686f6e616eULL);
  std::normal_distribution<double> normal;
  auto inverseIteration = [&](int steps) {
    // A second random start also covers a nearly orthogonal warm start.
    for (int start = 0; start < 2; ++start) {
      Eigen::VectorXd v = *direction;
      if (start == 1) {
        for (Eigen::Index i = 0; i < n; ++i) v(i) = normal(generator);
      }
      for (int i = 0; i < steps; ++i) {
        Eigen::VectorXd solved = factor.solve(v);
        const double length = solved.stableNorm();
        if (!(length > 0.0) || !std::isfinite(length)) break;
        v = solved / length;
        const double value = v.dot(certificate * v);
        if (value < rayleigh) {
          rayleigh = value;
          *direction = v;
        }
      }
    }
  };

  // Near a stationary point the certificate has an approximate null vector.
  // An initial shift near zero is therefore usually much better than the
  // center of the Gershgorin bracket.
  double probe = -accuracy * 0.5;
  while (upper - lower > accuracy) {
    if (!(lower < probe && probe < upper)) probe = 0.5 * lower + 0.5 * upper;
    factor.setShift(-probe).factorize(certificate);
    if (factor.info() == Eigen::Success) {
      lower = probe;
      inverseIteration(3);
      upper = std::min(upper, rayleigh + guard);
      // Try to validate a current Rayleigh estimate in a single factorization.
      probe = upper - 0.75 * accuracy;
    } else {
      upper = probe;
      probe = 0.5 * lower + 0.5 * upper;
    }
  }
  *eigenvalue_lower_bound = lower - guard;
  // The bracket can converge using failed shifts alone. Recover a direction
  // from below the bracket when the best vector seen still lacks curvature.
  if (rayleigh >= -guard && *eigenvalue_lower_bound < -accuracy - guard) {
    factor.setShift(-lower + std::max(accuracy, guard)).factorize(certificate);
    if (factor.info() != Eigen::Success) return false;
    inverseIteration(24);
    if (rayleigh >= -guard) return false;
  }
  if (rayleigh >= -guard) direction->resize(0);
  return true;
}

}  // namespace

bool minimumEigenpair(const Eigen::SparseMatrix<double>& certificate, double target_shift,
                      double* lower_bound, Eigen::VectorXd* direction) {
  // Dense decomposition is inexpensive for small components. Keep large
  // graph certificates sparse throughout the staircase.
  if (certificate.rows() > 192) {
    return sparseCertificateEigenpair(certificate, target_shift, lower_bound, direction);
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen{Eigen::MatrixXd(certificate)};
  if (eigen.info() != Eigen::Success) return false;
  const double guard = 32.0 * std::numeric_limits<double>::epsilon() *
                       eigen.eigenvalues().cwiseAbs().maxCoeff();
  *lower_bound = eigen.eigenvalues()(0) - guard;
  if (eigen.eigenvalues()(0) < -guard) *direction = eigen.eigenvectors().col(0);
  else direction->resize(0);
  return true;
}

}  // namespace shonan_internal

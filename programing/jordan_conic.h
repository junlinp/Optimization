#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

// Solve min c'x subject to A x = b and x in a product of symmetric cones.
// Semidefinite blocks use svec: diagonal entries followed by upper-triangular
// entries (column by column), with off-diagonals multiplied by sqrt(2).
namespace jordan {
enum class ConeType { Nonnegative, Lorentz, Semidefinite };
struct ConeBlock {
  ConeType type;
  int size;  // Vector dimension for nonnegative/Lorentz; matrix order for SDP.
};
enum class Status { Optimal, MaxIterations, NumericalFailure, InvalidInput };
// These options select factorization only; every cone uses the same Jordan iteration.
// Auto selects sparse normal equations for LPs and dense KKT otherwise.
enum class Backend { Auto, DenseJordan, SparseLP };
struct Options {
  int max_iterations = 150;
  double tolerance = 1e-8;
  Backend backend = Backend::Auto;
};
struct Result {
  Status status = Status::InvalidInput;
  Backend backend = Backend::Auto;
  Eigen::VectorXd x, y, s;
  int iterations = 0;
  double primal_residual = 0, dual_residual = 0, gap = 0;
};
int Dimension(const std::vector<ConeBlock>& blocks);
Eigen::VectorXd Identity(const std::vector<ConeBlock>& blocks);
Eigen::VectorXd Product(const std::vector<ConeBlock>& blocks,
                        const Eigen::VectorXd& x, const Eigen::VectorXd& y);
Result Solve(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
             const Eigen::VectorXd& c, const std::vector<ConeBlock>& blocks,
             const Options& options = {});
// Keeps LP input sparse. DenseJordan explicitly converts A to dense.
Result SolveSparse(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
                   const Eigen::VectorXd& c, const std::vector<ConeBlock>& blocks,
                   const Options& options = {});
}  // namespace jordan

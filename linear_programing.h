#include <Eigen/Dense>
#include <vector>
void LPSolver(const Eigen::VectorXd& c, const Eigen::MatrixXd& A,
              const Eigen::VectorXd& b, Eigen::VectorXd& x);
/**
 *  solve the problem
 *  min tr(C * X)
 *  s.t tr(A_i * X) = b_i for i = 0... m
 *      X >= 0  for X is a semidefinite cone
 */
void SymmetricSolver(const Eigen::MatrixXd C, const std::vector<Eigen::MatrixXd>& A,
                     const Eigen::VectorXd& b, Eigen::MatrixXd& x);
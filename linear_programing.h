#include <Eigen/Dense>
#include <vector>
void LPSolver(const Eigen::VectorXd& c, const Eigen::MatrixXd& A,
              const Eigen::VectorXd& b, Eigen::VectorXd& x);
/**
 * @brief solve the LP
 * 
 * min c^Tx
 * s.t Ax = b
 *     x >= 0
 **/
void LPSolver2(const Eigen::VectorXd& c, const Eigen::MatrixXd& A,
               const Eigen::VectorXd& b, Eigen::VectorXd& x);
/**
 *  solve the problem
 *  min tr(C * X)
 *  s.t tr(A_i * X) = b_i for i = 0... m
 *      X >= 0  for X is a semidefinite cone
 */
void SymmetricSolver(const Eigen::MatrixXd C,
                     const std::vector<Eigen::MatrixXd>& A,
                     const Eigen::VectorXd& b, Eigen::MatrixXd& x);

/**
 * @brief solve the SDP problem
 *
 *  min tr(C * X)
 *  s.t tr(A_i * X) = b_i for i = 0... m
 *      X >= 0  for X is a semidefinite cone
 *
 * bibliography:
 *      <Full Newton Step Interior Point Method for Linear Complementarity
 * Problem Over Symmetric Cones> Andrii Berdnikov
 * @param C
 * @param A
 * @param b
 * @param x
 */
void SymmetricSolver2(const Eigen::MatrixXd C,
                      const std::vector<Eigen::MatrixXd>& A,
                      const Eigen::VectorXd& b, Eigen::MatrixXd& x);

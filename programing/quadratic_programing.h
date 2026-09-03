#ifndef PROGRAMING_QUADRATIC_PROGRAMING_H_
#define PROGRAMING_QUADRATIC_PROGRAMING_H_

#include <Eigen/Dense>

/**
 * @brief Solve the convex QP
 *
 *   min  0.5 x^T H x + g^T x
 *   s.t. A x = b
 *        C x <= d
 *
 * via a primal-dual interior-point method. H must be symmetric positive
 * semi-definite. Either constraint block may be empty (zero rows) to omit
 * that constraint type; box bounds lb <= x <= ub can be encoded as
 * C = [I; -I], d = [ub; -lb].
 *
 * @param H symmetric PSD n x n objective matrix
 * @param g n-vector linear objective term
 * @param A m_eq x n equality constraint matrix (may have 0 rows)
 * @param b m_eq-vector equality constraint RHS
 * @param C m_ineq x n inequality constraint matrix (may have 0 rows)
 * @param d m_ineq-vector inequality constraint RHS
 * @param x on return, the solution (also used to size the problem)
 * @return true if the residuals and duality gap converged within tolerance
 */
bool QPSolver(const Eigen::MatrixXd& H, const Eigen::VectorXd& g,
              const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
              const Eigen::MatrixXd& C, const Eigen::VectorXd& d,
              Eigen::VectorXd& x);

#endif  // PROGRAMING_QUADRATIC_PROGRAMING_H_

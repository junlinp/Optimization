#include "gtest/gtest.h"
#include "Eigen/Dense"
#include "quadratic_programing.h"

TEST(QP, Unconstrained) {
  // min x1^2 + x2^2 - 4x1 - 6x2
  Eigen::MatrixXd H(2, 2);
  H << 2, 0, 0, 2;
  Eigen::VectorXd g(2);
  g << -4, -6;
  Eigen::MatrixXd A(0, 2);
  Eigen::VectorXd b(0);
  Eigen::MatrixXd C(0, 2);
  Eigen::VectorXd d(0);
  Eigen::VectorXd x;

  EXPECT_TRUE(QPSolver(H, g, A, b, C, d, x));
  // x* = [2, 3]
  EXPECT_NEAR(x(0), 2.0, 1e-6);
  EXPECT_NEAR(x(1), 3.0, 1e-6);
}

TEST(QP, EqualityConstrained) {
  // min 0.5*(x1^2 + x2^2) s.t. x1 + x2 = 4
  Eigen::MatrixXd H = Eigen::MatrixXd::Identity(2, 2);
  Eigen::VectorXd g = Eigen::VectorXd::Zero(2);
  Eigen::MatrixXd A(1, 2);
  A << 1, 1;
  Eigen::VectorXd b(1);
  b << 4;
  Eigen::MatrixXd C(0, 2);
  Eigen::VectorXd d(0);
  Eigen::VectorXd x;

  EXPECT_TRUE(QPSolver(H, g, A, b, C, d, x));
  // x* = [2, 2]
  EXPECT_NEAR(x(0), 2.0, 1e-6);
  EXPECT_NEAR(x(1), 2.0, 1e-6);
  EXPECT_NEAR(0.5 * x.dot(H * x) + g.dot(x), 4.0, 1e-6);
}

TEST(QP, InequalityActive) {
  // min 0.5*||x - p||^2 s.t. x1 <= 3, with p = [5, 1].
  // Unconstrained optimum is [5, 1]; the bound on x1 is active.
  Eigen::MatrixXd H = Eigen::MatrixXd::Identity(2, 2);
  Eigen::VectorXd g(2);
  g << -5, -1;
  Eigen::MatrixXd A(0, 2);
  Eigen::VectorXd b(0);
  Eigen::MatrixXd C(1, 2);
  C << 1, 0;
  Eigen::VectorXd d(1);
  d << 3;
  Eigen::VectorXd x;

  EXPECT_TRUE(QPSolver(H, g, A, b, C, d, x));
  // x* = [3, 1]
  EXPECT_NEAR(x(0), 3.0, 1e-6);
  EXPECT_NEAR(x(1), 1.0, 1e-6);
}

TEST(QP, EqualityAndInequalityActive) {
  // min 0.5*(x1^2 + x2^2) s.t. x1 + x2 = 4, x1 <= 1.
  // Without the inequality, x* would be [2, 2]; the bound on x1 is active.
  Eigen::MatrixXd H = Eigen::MatrixXd::Identity(2, 2);
  Eigen::VectorXd g = Eigen::VectorXd::Zero(2);
  Eigen::MatrixXd A(1, 2);
  A << 1, 1;
  Eigen::VectorXd b(1);
  b << 4;
  Eigen::MatrixXd C(1, 2);
  C << 1, 0;
  Eigen::VectorXd d(1);
  d << 1;
  Eigen::VectorXd x;

  EXPECT_TRUE(QPSolver(H, g, A, b, C, d, x));
  // x* = [1, 3]
  EXPECT_NEAR(x(0), 1.0, 1e-6);
  EXPECT_NEAR(x(1), 3.0, 1e-6);
}

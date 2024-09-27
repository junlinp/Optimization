#include <gtest/gtest.h>
#include "lp/linear_programing.h"

TEST(LP, Basic) {
  Eigen::VectorXd c(4);
  c << 1, -1, 0, 0;
  Eigen::MatrixXd A(2, 4);
  A << 10, -7, -1, 0, 1, 0.5, 0, 1;
  Eigen::VectorXd b(2);
  b << 5.0, 3.0;
  Eigen::VectorXd x;
  LPSolver(c, A, b, x);
  std::cout << "x : " << x << std::endl;
  std::cout << "A * x - b : " << b - A * x << std::endl;
}

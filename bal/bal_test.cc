#include "evaluate.h"
#include "gtest/gtest.h"
/*
TEST(Project_Function, simple_case) {
  // clang-format off
  double camera[9] = {
      0.0157415, -0.0127909, -0.00440085,
     -0.0340938,  -0.107514, 1.12022,
      399.752,    -3.17706e-07, 5.88205e-13,
  };
  // clang-format on
  double point[3] = {-0.612, 0.571759, -1.84708};
  double uv[2] = {-332.65, 262.09};

  ProjectFunction p(uv[0], uv[1]);
  double residual[2];
  p(camera, point, residual);
  EXPECT_NEAR(residual[0], 9.01928, 1e-5);
  EXPECT_NEAR(residual[1], -11.26312, 1e-5);
}
*/

TEST(PLUS, JETD) {
  JETD<2> xy[2];
  xy[0] = JETD<2>(1.0, 0);
  xy[1] = JETD<2>(2.0, 1);

  JETD<2> error = xy[0] + xy[1];

  EXPECT_NEAR(3.0, error.value(), 1e-7);

  EXPECT_NEAR(1.0, error.Gradient()(0), 1e-7);
  EXPECT_NEAR(1.0, error.Gradient()(1), 1e-7);
}

TEST(MINUS, JETD) {
  JETD<2> xy[2];
  xy[0] = JETD<2>(1.0, 0);
  xy[1] = JETD<2>(2.0, 1);

  JETD<2> error = xy[0] - xy[1];

  EXPECT_NEAR(-1.0, error.value(), 1e-7);

  EXPECT_NEAR(1.0, error.Gradient()(0), 1e-7);
  EXPECT_NEAR(-1.0, error.Gradient()(1), 1e-7);
}

TEST(MULTIPLE, JETD) {
  JETD<2> xy[2];
  xy[0] = JETD<2>(1.0, 0);
  xy[1] = JETD<2>(2.0, 1);

  JETD<2> error = xy[0] * xy[1];

  EXPECT_NEAR(2.0, error.value(), 1e-7);

  EXPECT_NEAR(2.0, error.Gradient()(0), 1e-7);
  EXPECT_NEAR(1.0, error.Gradient()(1), 1e-7);
}


TEST(DIVSION, JETD) {
  JETD<2> xy[2];
  xy[0] = JETD<2>(1.0, 0);
  xy[1] = JETD<2>(2.0, 1);

  JETD<2> error = xy[0] / xy[1];

  EXPECT_NEAR(0.5, error.value(), 1e-7);

  EXPECT_NEAR(0.5, error.Gradient()(0), 1e-7);
  EXPECT_NEAR(-0.25, error.Gradient()(1), 1e-7);
}
template<class T>
void functor(T* input, T* output) {
    std::cout << "input " << input->value() << std::endl;
    JETD<1> plus = *input + JETD<1>{1.0};
    std::cout << "plus " << plus.value() << std::endl;
    *output = sqrt(plus / 2.0);
    std::cout << "output : " << output->value() << std::endl;
}
TEST(Simple_Case,  JETD) {
  JETD<1> x{3.0, 0};
  JETD<1> y;

  functor(&x, &y);
  std::cout << x.value() << std::endl;

  EXPECT_NEAR(std::sqrt(2.0), y.value(), 1e-7);
  EXPECT_NEAR(0.25, y.Gradient()(0), 1e-7);
}

/*
TEST(Project_Function, JETD) {
  JETD<12> param[12];
  double p[12] = {0.0157415,   -0.0127909, -0.00440085, -0.0340938,
                  -0.107514,   1.12022,    399.752,     -3.17706e-07,
                  5.88205e-13, -0.612,     0.571759,    -1.84708};
  JETD<12> param_plus[12];
  JETD<12> param_sub[12];
  size_t check_index = 11;
  for (size_t i = 0; i < 12; i++) {
    param[i] = JETD<12>(p[i], i);
    if (i == check_index) {
      param_plus[i] =JETD<12>(p[i] + 1e-9, i);
      param_sub[i] = JETD<12>(p[i] - 1e-9, i);
    } else {
      param_plus[i] =JETD<12>(p[i], i);
      param_sub[i] = JETD<12>(p[i], i);
    }
  }
  JETD<12> residual[2];
  ProjectFunction functor(-332.65, 262.09);

  functor(param, param + 9, residual);
  ASSERT_NEAR(residual[0].value(), 9.01928, 1e-5);
  ASSERT_NEAR(residual[1].value(), -11.26312, 1e-5);

  JETD<12> residual_plus[2];
  JETD<12> residual_sub[2];
  functor(param_plus, param_plus + 9, residual_plus);
  functor(param_sub, param_sub + 9, residual_sub);

  double checkout_gradient =
      (residual_plus[1].value() - residual_sub[1].value()) / 2e-9;
  std::cout << "checkout_gradient : " << checkout_gradient << std::endl;
  std::cout << "auto gradient : " << residual[1].Gradient()(check_index)
            << std::endl;

  Eigen::MatrixXd jacobian(2, 12);
  Eigen::VectorXd error(2);
  error(0) = residual[0].value();
  error(1) = residual[1].value();
  jacobian.row(0) = residual[0].Gradient();
  jacobian.row(1) = residual[1].Gradient();

  size_t iterator = 0;
  double lambda = 1;
  while (iterator++ < 1) {
    Eigen::MatrixXd jacobian(2, 12);
    Eigen::VectorXd error(2);
    error(0) = residual[0].value();
    error(1) = residual[1].value();
    jacobian.row(0) = residual[0].Gradient();
    jacobian.row(1) = residual[1].Gradient();
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(12, 12);
    for (int i = 0; i < 12; i++) {
      D(i, i) = jacobian.col(i).norm();
    }
    std::cout << "jacobian : " << jacobian << std::endl;
    Eigen::VectorXd Step =
        -(jacobian.transpose() * jacobian + lambda * D.transpose() * D)
             .inverse() *
        jacobian.transpose() * error;
    std::cout << "Step : " << Step << std::endl;
    for (int i = 0; i < 12; i++) {
      std::cout << "Before Update : " << param[i].value() << std::endl;
    }
    for (int i = 0; i < 12; i++) {
      param[i] += Step[i];
    }
    functor(param, param + 9, residual);
    std::cout << "Residual[0] : " << residual[0].value() << std::endl;
    std::cout << "Residual[1] : " << residual[1].value() << std::endl;
    double a = residual[0].value(), b = residual[1].value();
    std::cout << "RMS : " << 0.5 * ((a * a) + (b * b)) << std::endl;
  }
  for (int i = 0; i < 12; i++) {
    std::cout << "After Update : " << param[i].value() << std::endl;
  }
}
*/

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
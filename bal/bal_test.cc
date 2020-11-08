#include "gtest/gtest.h"
#include "evaluate.h"

TEST(Project_Function, simple_case) {
  double camera[9] = {
      0.0157415, -0.0127909, -0.00440085,  -0.0340938,  -0.107514,
      1.12022,   399.752,    -3.17706e-07, 5.88205e-13,
  };
  double point[3] = {-0.612, 0.571759, -1.84708};
  double uv[2] = {-332.65, 262.09};

  ProjectFunction p(uv[0], uv[1]);
  double residual[2];
  p(camera, point, residual);
  std::cout << "Residual[0] : " << residual[0] << std::endl;
  std::cout << "Residual[1] : " << residual[1] << std::endl;
}

TEST(Project_Function, Jet) {
    Jet<12> param[12];
    double p[12] = {0.0157415,   -0.0127909, -0.00440085, -0.0340938,
                    -0.107514,   1.12022,    399.752,     -3.17706e-07,
                    5.88205e-13, -0.612,     0.571759,    -1.84708};
    for(int i = 0; i < 12; i++) {
        param[i] = Jet<12>(p[i], i);
    }
    Jet<12> residual[2];
    ProjectFunction functor(-332.65, 262.09);

    functor(param, param + 9, residual);

    std::cout << "Residual[0] : " << residual[0].value() << std::endl;
    std::cout << "Residual[1] : " << residual[1].value() << std::endl;

    Jet<12> param_plus[12];
    Jet<12> param_sub[12];
    size_t check_index = 11;
    for (size_t i = 0; i < 12; i++) {
      param_plus[i] = param[i];
      param_sub[i] = param[i];
      if (i == check_index) {
        param_plus[i] += 1e-5;
        param_sub[i] -= 1e-5;
      }
    }
    Jet<12> residual_plus[2];
    Jet<12> residual_sub[2];
    functor(param_plus, param_plus + 9, residual_plus);
    functor(param_sub, param_sub + 9, residual_sub);

    double checkout_gradient = (residual_plus[1].value() - residual_sub[1].value()) / 2e-5;
    std::cout << "checkout_gradient : " << checkout_gradient << std::endl;
    std::cout << "auto gradient : " << residual[1].Gradient()(check_index) << std::endl;

    Eigen::MatrixXd jacobian(2, 12);
    Eigen::VectorXd error(2);
    error(0) = residual[0].value();
    error(1) = residual[1].value();
    jacobian.row(0) = residual[0].Gradient();
    jacobian.row(1) = residual[1].Gradient();
    
    size_t iterator = 0;
    double lambda = 1;
    while(iterator++ < 1) {
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

int main() {
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
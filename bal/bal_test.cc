#include <ceres/autodiff_cost_function.h>
#include <ceres/solver.h>

#include "ceres/problem.h"
#include "cost_function_auto.h"
#include "evaluate.h"
#include "problem.h"
#include "gtest/gtest.h"

double CostValueFromResidual(double* values) {
  return 0.5 * (values[0] * values[0] + values[1] * values[1] + values[2] * values[2]);
}

TEST(Project_Function, simple_case) {
  // clang-format off
  double camera[9] = {
      0.0157415, -0.0127909, -0.00440085,
     -0.0340938,  -0.107514, 1.12022,
      399.752,    -3.17706e-07, 5.88205e-13,
  };
  // clang-format on
  double point[3] = {-0.612, 0.571759, -1.84708};
  // double uv[2] = {-332.65, 262.09};
  double uv[2] = {332.65, -262.09};

  ProjectFunction p(uv[0], uv[1]);
  double residual[2];
  p(camera, point, residual);
  EXPECT_NEAR(residual[0] * residual[0], 9.01928 * 9.01928, 1e-3);
  EXPECT_NEAR(residual[1] * residual[1], 11.26312 * 11.26312, 1e-3);
}

TEST(Surrogate_Function, simple_case) {
  // clang-format off
  double camera[9] = {
      0.0157415, -0.0127909, -0.00440085,
     -0.0340938,  -0.107514, 1.12022,
      399.752,    -3.17706e-07, 5.88205e-13,
  };
  // clang-format on
  double point[3] = {-0.612, 0.571759, -1.84708};

  double condition_camera[9] = {
      0.0157415, -0.0127909, -0.00440085,  -0.0340938,  -0.107514,
      1.12022,   399.752,    -3.17706e-07, 5.88205e-13,
  };
  // clang-format on
  double condition_point[3] = {-0.612, 0.571759, -1.84708};
  // double uv[2] = {-332.65, 262.09};
  double uv[2] = {332.65, -262.09};

  CameraSurrogateCostFunction camera_surrogate_cost_function(
      condition_camera, condition_point, uv[0], uv[1]);
  LandmarkSurrogatecostFunction landmark_surrogate_cost_function(
      condition_camera, condition_point, uv[0], uv[1]);
  RayCostFunction ray_cost_function(uv[0], uv[1]);

  double camera_residual[3];
  double landmark_residual[3];
  double ray_residual[3];

  camera_surrogate_cost_function(camera, camera_residual);
  landmark_surrogate_cost_function(point, landmark_residual);
  ray_cost_function(camera, point, ray_residual);

  std::cout << camera_residual[0] << ", " << camera_residual[1] << ", "
            << camera_residual[2] << std::endl;
  std::cout << landmark_residual[0] << ", " << landmark_residual[1] << ", "
            << landmark_residual[2] << std::endl;
  std::cout << ray_residual[0] << ", " << ray_residual[1] << ", "
            << ray_residual[2] << std::endl;

  Eigen::Vector3d camera_residual_vector(camera_residual);
  Eigen::Vector3d landmark_residual_vector(landmark_residual);
  Eigen::Vector3d ray_residual_vector(ray_residual);

  EXPECT_NEAR(ray_residual_vector.squaredNorm(),
              camera_residual_vector.squaredNorm() +
                  landmark_residual_vector.squaredNorm(),
              1e-6);

  ceres::Problem problem;
  problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<CameraSurrogateCostFunction, 3, 9>(
          new CameraSurrogateCostFunction(condition_camera, condition_point,
                                          uv[0], uv[1])),
      nullptr, camera);
  problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<LandmarkSurrogatecostFunction, 3, 3>(
          new LandmarkSurrogatecostFunction(condition_camera, condition_point,
                                         uv[0], uv[1])),
      nullptr, point);

  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;


  camera_surrogate_cost_function(camera, camera_residual);
  landmark_surrogate_cost_function(point, landmark_residual);
  ray_cost_function(camera, point, ray_residual);
  camera_residual_vector = Eigen::Vector3d(camera_residual);
  landmark_residual_vector = Eigen::Vector3d(landmark_residual);
  ray_residual_vector = Eigen::Vector3d(ray_residual);

  std::cout << camera_residual[0] << ", " << camera_residual[1] << ", "
            << camera_residual[2] << std::endl;
  std::cout << landmark_residual[0] << ", " << landmark_residual[1] << ", "
            << landmark_residual[2] << std::endl;
  std::cout << ray_residual[0] << ", " << ray_residual[1] << ", "
            << ray_residual[2] << std::endl;

  EXPECT_LT(ray_residual_vector.squaredNorm(),
              camera_residual_vector.squaredNorm() +
                  landmark_residual_vector.squaredNorm()
              );
}

TEST(Surrogate_Function, partition_camera_optimization) {
  // clang-format off
  double camera[9] = {
      0.0157415, -0.0127909, -0.00440085,
     -0.0340938,  -0.107514, 1.12022,
      399.752,    -3.17706e-07, 5.88205e-13,
  };
  // clang-format on
  double point[3] = {-0.612, 0.571759, -1.84708};

  double condition_camera[9] = {
      0.0157415, -0.0127909, -0.00440085,  -0.0340938,  -0.107514,
      1.12022,   399.752,    -3.17706e-07, 5.88205e-13,
  };
  // clang-format on
  double condition_point[3] = {-0.612, 0.571759, -1.84708};
  // double uv[2] = {-332.65, 262.09};
  double uv[2] = {332.65, -262.09};

  CameraSurrogateCostFunction camera_surrogate_cost_function(
      condition_camera, condition_point, uv[0], uv[1]);
  double camera_residual[3];

  camera_surrogate_cost_function(camera, camera_residual);
  
  double x_under_x_camera_error = CostValueFromResidual(camera_residual);

  ceres::Problem problem;
  problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<CameraSurrogateCostFunction, 3, 9>(
          new CameraSurrogateCostFunction(condition_camera, condition_point,
                                          uv[0], uv[1])),
      nullptr, camera);
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
  camera_surrogate_cost_function(camera, camera_residual);

  CameraSurrogateCostFunction camera_surrogate_cost_function_new_x(
      camera, condition_point, uv[0], uv[1]);

  camera_surrogate_cost_function_new_x(camera, camera_residual);
  double x_under_new_x_camera_error = CostValueFromResidual(camera_residual);

  EXPECT_LE(x_under_new_x_camera_error, x_under_x_camera_error);
}

TEST(Surrogate_Function, partition_point_optimization) {
  // clang-format off
  double camera[9] = {
      0.0201406,-0.0481816,-0.00528055,0.119713,-0.0600754,1.62656,412.015,2.19135e-07,-4.22647e-13
  };
  // clang-format on
  double point[3] = {0.991267,-0.0529383,-3.60759};

  double condition_camera[9] = {
      0.0201406,-0.0481816,-0.00528055,0.119713,-0.0600754,1.62656,412.015,2.19135e-07,-4.22647e-13
  };
  // clang-format on
  double condition_point[3] = {0.991267,-0.0529383,-3.60759};
  // double uv[2] = {-332.65, 262.09};
  double uv[2] = {-277.65, 10.07};

  LandmarkSurrogatecostFunction landmark_surrogate_cost_function(
      condition_camera, condition_point, uv[0], uv[1]);

  ceres::Problem problem;
  problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<LandmarkSurrogatecostFunction, 3, 9>(
          new LandmarkSurrogatecostFunction(condition_camera, condition_point,
                                            uv[0], uv[1])),
      nullptr, point);
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  double point_residual[3];
  landmark_surrogate_cost_function(point, point_residual);
  double x_under_x_camera_error = CostValueFromResidual(point_residual);

  LandmarkSurrogatecostFunction landmark_new_cost(
      condition_camera, point, uv[0], uv[1]);

  landmark_new_cost(point, point_residual);
  double x_under_new_x_camera_error = CostValueFromResidual(point_residual);
  std::cout << "new : " << x_under_new_x_camera_error << std::endl;
  std::cout << "pre : " << x_under_x_camera_error << std::endl;
  EXPECT_LE(x_under_new_x_camera_error, x_under_x_camera_error);
}

TEST(CameraProject, Convert) {
  std::array<double, 9> camera = {0.0201406, -0.0481816,  -0.00528055,
                      0.119713,  -0.0600754,  1.62656,
                      412.015,   2.19135e-07, -4.22647e-13};

  auto rotation_matrix = CameraParam::ConvertLieAlgrebaToRotationMatrix(camera);

  auto convert_back = CameraParam::Project(rotation_matrix);


  for(int i = 0; i < camera.size(); i++) {
    EXPECT_NEAR(camera[i], convert_back[i], 1e-6);
  }
}
// TEST(PLUS, JETD) {
//   JETD<2> xy[2];
//   xy[0] = JETD<2>(1.0, 0);
//   xy[1] = JETD<2>(2.0, 1);

//   JETD<2> error = xy[0] + xy[1];

//   EXPECT_NEAR(3.0, error.value(), 1e-7);

//   EXPECT_NEAR(1.0, error.Gradient()(0), 1e-7);
//   EXPECT_NEAR(1.0, error.Gradient()(1), 1e-7);
// }

// TEST(MINUS, JETD) {
//   JETD<2> xy[2];
//   xy[0] = JETD<2>(1.0, 0);
//   xy[1] = JETD<2>(2.0, 1);

//   JETD<2> error = xy[0] - xy[1];

//   EXPECT_NEAR(-1.0, error.value(), 1e-7);

//   EXPECT_NEAR(1.0, error.Gradient()(0), 1e-7);
//   EXPECT_NEAR(-1.0, error.Gradient()(1), 1e-7);
// }

// TEST(MULTIPLE, JETD) {
//   JETD<2> xy[2];
//   xy[0] = JETD<2>(1.0, 0);
//   xy[1] = JETD<2>(2.0, 1);

//   JETD<2> error = xy[0] * xy[1];

//   EXPECT_NEAR(2.0, error.value(), 1e-7);

//   EXPECT_NEAR(2.0, error.Gradient()(0), 1e-7);
//   EXPECT_NEAR(1.0, error.Gradient()(1), 1e-7);
// }

// TEST(DIVSION, JETD) {
//   JETD<2> xy[2];
//   xy[0] = JETD<2>(1.0, 0);
//   xy[1] = JETD<2>(2.0, 1);

//   JETD<2> error = xy[0] / xy[1];

//   EXPECT_NEAR(0.5, error.value(), 1e-7);

//   EXPECT_NEAR(0.5, error.Gradient()(0), 1e-7);
//   EXPECT_NEAR(-0.25, error.Gradient()(1), 1e-7);
// }

// TEST(JETD, Sqrt) {
//   JETD<1> x{4, 0};

//   JETD<1> y = sqrt(x);

//   EXPECT_NEAR(y.value(), 2.0, 1e-7);
//   EXPECT_NEAR(y.Gradient()(0), 0.25, 1e-7);
// }

// template<class T>
// void functor(T* input, T* output) {
//     auto plus = *input + 1.0;
//     *output = sqrt(plus / 2.0);
// }

// TEST(Simple_Case,  JETD) {
//   JETD<1> x{3.0, 0};
//   JETD<1> y;

//   functor(&x, &y);
//   std::cout << x.value() << std::endl;

//   EXPECT_NEAR(std::sqrt(2.0), y.value(), 1e-7);
//   EXPECT_NEAR(0.25 / std::sqrt(2), y.Gradient()(0), 1e-7);
// }

// TEST(JETD, Gradient_Checker) {
//   double x = 3.0;
//   auto l = [](auto* input, auto* output) {
//     functor(input, output);
//   };
//   bool check = GradientCheck<1, 1>(l, &x, 1e-7);
//   EXPECT_TRUE(check);
// }

/*
TEST(JETD, Gradient_Checker_Project_Function) {
  double p[12] = {0.0157415,   -0.0127909, -0.00440085, -0.0340938,
                  -0.107514,   1.12022,    399.752,     -3.17706e-07,
                  5.88205e-13, -0.612,     0.571759,    -1.84708};
  auto lambda_functor = [](auto* input, auto* output) {
    ProjectFunction functor(-332.65, 262.09);
    functor(input, input + 9, output);
  };
  bool check = GradientCheck<2, 12>(lambda_functor, p, 1e-5);

  EXPECT_TRUE(check);

}

TEST(Project_Function, JETD) {
  JETD<12> param[12];
  double p[12] = {0.0157415,   -0.0127909, -0.00440085, -0.0340938,
                  -0.107514,   1.12022,    399.752,     -3.17706e-07,
                  5.88205e-13, -0.612,     0.571759,    -1.84708};
  JETD<12> param_plus[12];
  JETD<12> param_sub[12];
  size_t check_index = 2;
  for (size_t i = 0; i < 12; i++) {
    param[i] = JETD<12>(p[i], i);
    if (i == check_index) {
      param_plus[i] =JETD<12>(p[i] + 1e-4, i);
      param_sub[i] = JETD<12>(p[i] - 1e-4, i);
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
      (residual_plus[1].value() - residual_sub[1].value()) / 2e-4;
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
  while (iterator++ < 2) {
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
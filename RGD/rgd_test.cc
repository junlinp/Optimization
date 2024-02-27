#include "rgd.h"

#include "gtest/gtest.h"
#include "manifold.h"
#include "so3_cost_function_interface.h"
TEST(RGD, Basic) {
  struct BasicCostFunction : public SO3CostFunctionInterface {
   private:
    Eigen::Matrix3d target;

   public:
    BasicCostFunction() { target << 1, 0, 0, 0, 0, 1, 0, -1, 0; }
    double Evaluate(const std::vector<SO3Manifold::Vector> &x) const override {
      double ret = 0.0;
      for (const auto &item : x) {
        Eigen::Map<const Eigen::Matrix3d> t(item.data());
        Eigen::Matrix3d error =
            (target.transpose() * t) - Eigen::Matrix3d::Identity();
        ret += error.array().square().sum();
      }
      return ret;
    }
    std::vector<SO3Manifold::Vector> Jacobian(
        const std::vector<SO3Manifold::Vector> &x) const override {
      Eigen::Matrix3d A = target.transpose().eval();
      Eigen::Matrix<double, 9, 9> G;
      G.setZero();
      G.block<1, 3>(0, 0) = A.row(0);
      G.block<1, 3>(1, 3) = A.row(1);
      G.block<1, 3>(2, 6) = A.row(2);
      G.block<1, 3>(3, 0) = A.row(0);
      G.block<1, 3>(4, 3) = A.row(1);
      G.block<1, 3>(5, 6) = A.row(2);
      G.block<1, 3>(6, 0) = A.row(0);
      G.block<1, 3>(7, 3) = A.row(1);
      G.block<1, 3>(8, 6) = A.row(2);
      std::cout << " G : " << G << std::endl;
      std::vector<SO3Manifold::Vector> res;

      Eigen::Matrix<double, 3, 3> row_major_identity =
          Eigen::Matrix3d::Identity();
      Eigen::Map<Eigen::Matrix<double, 9, 1>> vector_identity(
          row_major_identity.data());
      for (const auto &item : x) {
        std::cout << "Error : " << (G * item - vector_identity).squaredNorm()
                  << std::endl;
        Eigen::Matrix<double, 9, 1> jacobian =
            2.0 * G.transpose() * (G * item - vector_identity);
        std::cout << "jacobian : " << jacobian << std::endl;

        res.push_back(jacobian);
      }
      return res;
    }
  };

  BasicCostFunction function;
  SO3Manifold::Vector x0;
  double theta = 3.14 * 0.4;
  x0 << 1.0, 0.0, 0.0, 0.0, std::cos(theta), -std::sin(theta), 0.0,
      std::sin(theta), std::cos(theta);
  std::cout << "x0 " << x0 << std::endl;
  std::vector<SO3Manifold::Vector> x_init{x0};

  rgd(function, &x_init);
  Eigen::Map<Eigen::Matrix3d> matrix_solution(x_init[0].data());
  std::cout << "Solution : " << matrix_solution << std::endl;
  std::cout << "Rotation Matrix ? "
            << matrix_solution.transpose() * matrix_solution << std::endl;
}

class CostFunction {
 public:
  using ResidualVector = Eigen::Matrix<double, 3, 1>;
  using JacobianMatrix = Eigen::Matrix<double, 3, 3>;

  template <class Manifold>
  bool Evaluate(const Manifold &parameters, ResidualVector *residuals,
                JacobianMatrix *jacobian_matrix) const {
    Eigen::Vector3d param = parameters.AmbientVector();
    Eigen::Matrix3d A;
    A << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    Eigen::Vector3d b;
    b << 1.0, 2.0, 3.0;

    *residuals = A * param - b;
    if (jacobian_matrix != nullptr) {
      *jacobian_matrix = A;
    }
    return true;
  }
};

TEST(LeastQuaresRiemannGredientDescentLinearSearch, EuclideanManifold) {
  CostFunction functor;
  EuclideanManifold<3> manifold;
  Eigen::Vector3d step;
  bool res = LeastQuaresRiemannGredientDescentLinearSearch(functor, manifold, step);
  EXPECT_TRUE(res);
  Eigen::Vector3d target;
  target << 2, -1, 3;

  EXPECT_LT(std::abs(step.cross(target).norm()), 1e-6);
}

TEST(LeastQuaresRiemannGredientDescentLinearSearch, TwoProductManifold) {
  CostFunction functor;
  auto manifold = CreateManifold(EuclideanManifold<2>(), EuclideanManifold<1>());
  Eigen::Vector3d step;
  bool res = LeastQuaresRiemannGredientDescentLinearSearch(functor, manifold, step);
  EXPECT_TRUE(res);
  Eigen::Vector3d target;
  target << 2, -1, 3;

  EXPECT_LT(std::abs(step.cross(target).norm()), 1e-6);
}

TEST(LeastQuaresRiemannGredientDescentLinearSearch, ThreeProductManifold) {
  CostFunction functor;
  auto manifold =
      CreateManifold(EuclideanManifold<1>(), EuclideanManifold<1>(),
                     EuclideanManifold<1>());
  Eigen::Vector3d step;
  bool res = LeastQuaresRiemannGredientDescentLinearSearch(functor, manifold, step);
  EXPECT_TRUE(res);
  Eigen::Vector3d target;
  target << 2, -1, 3;

  EXPECT_LT(std::abs(step.cross(target).norm()), 1e-6);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
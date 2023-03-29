#include "rgd.h"
#include "so3_cost_function_interface.h"
#include "gtest/gtest.h"

TEST(RGD, Basic) {
  struct BasicCostFunction : public SO3CostFunctionInterface {
  private:
    Eigen::Matrix3d target;
  public:
    BasicCostFunction() { target << 1, 0, 0, 0, 0, 1, 0, 1, 0; }
    double Evaluate(const std::vector<Eigen::Matrix3d> &x) const override {
      double ret = 0.0;
      for (const auto &item : x) {
        Eigen::Matrix3d error =
            (target.transpose() * item) - Eigen::Matrix3d::Identity();
        ret += error.array().square().sum();
      }
      return ret;
    }
    std::vector<Eigen::Matrix3d>
    Jacobian(const std::vector<Eigen::Matrix3d> &x) const override {
        Eigen::Matrix3d A = target.transpose().eval();
        Eigen::Matrix<double, 9, 9> G;
        G.setZero();
        G.block<1, 3>(0, 0) = A.row(0);
        G.block<1, 3>(1, 3) = A.row(0);
        G.block<1, 3>(2, 6) = A.row(0);
        G.block<1, 3>(3, 0) = A.row(1);
        G.block<1, 3>(4, 3) = A.row(1);
        G.block<1, 3>(5, 6) = A.row(1);
        G.block<1, 3>(6, 0) = A.row(2);
        G.block<1, 3>(7, 3) = A.row(2);
        G.block<1, 3>(8, 6) = A.row(2);
        std::cout << " G : " << G << std::endl;
        std::vector<Eigen::Matrix3d> res;
        
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> row_major_identity = Eigen::Matrix3d::Identity();
        Eigen::Map<Eigen::Matrix<double, 9, 1>> vector_identity(row_major_identity.data());
        for (const auto& item : x) {
            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> row_major_x = item;
            Eigen::Map<Eigen::Matrix<double, 9, 1>> vector_x(row_major_x.data());

            Eigen::Matrix<double, 9, 1> jacobian = 2.0 * (G * vector_x - vector_identity).transpose() * G;
            std::cout << "jacobian : " << jacobian << std::endl;
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_row_major(jacobian.data());

            res.push_back(jacobian_row_major);
        }
        return res;
    }
  };

  BasicCostFunction function;
  Eigen::Matrix3d x0;
  x0 << 1.0, 0.0, 0.0,
        0.0, 0.707, -0.707,
        0.0, -0.707, 0.707;
  std::vector<Eigen::Matrix3d> x_init{x0};
  rgd(function, &x_init);
  std::cout << "Solution : " << x_init[0] << std::endl;
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
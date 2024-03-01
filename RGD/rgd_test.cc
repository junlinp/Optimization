#include "rgd.h"

#include "gtest/gtest.h"
#include "manifold.h"
#include "rgd_cost_function_interface.h"
#include "so3_cost_function_interface.h"

#include "ceres/jet.h"
/*
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
    Eigen::Vector3d param = parameters;
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
  Eigen::Vector3d manifold = Eigen::Vector3d::Zero();
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
*/
class SpecialEuclideanManifoldCostFunction : public RGDFirstOrderInterface {
 public:
  using ResidualVector = Eigen::Matrix<double, 9 + 3, 1>;
  using JacobianMatrix = Eigen::Matrix<double, 9 + 3, 9 + 3>;


  Eigen::Matrix3d rotation_L_, rotation_R_;
  Eigen::Vector3d translation_L_, translation_R_;

  SpecialEuclideanManifoldCostFunction(Eigen::Matrix3d rotation_L,
                                       Eigen::Matrix3d rotation_R,
                                       Eigen::Vector3d translation_L,
                                       Eigen::Vector3d translation_R)
      : rotation_L_(rotation_L),
        rotation_R_(rotation_R),
        translation_L_(translation_L),
        translation_R_(translation_R) {}

  template <class Manifold>
  bool Evaluate(const Manifold &parameters, ResidualVector *residuals,
                JacobianMatrix *jacobian_matrix) const {
    Eigen::VectorXd param = parameters;

    using JetType = ceres::Jet<double, 12>;
    ceres::Jet<double, 12> jet_parameters[12];
    // init
    for (int i = 0; i < 12; i++) {
      jet_parameters[i].a = param(i);
      jet_parameters[i].v.setZero();
      jet_parameters[i].v(i) = 1.0;
    }
    
    Eigen::Map<Eigen::Matrix<JetType, 3, 3>> R(jet_parameters);
    Eigen::Map<Eigen::Matrix<JetType, 3, 1>> t(jet_parameters + 9);

    Eigen::Matrix<JetType, 3, 3> res_R =
        rotation_L_.transpose() * R.transpose() * rotation_R_ * R;

    Eigen::Matrix<JetType, 3, 1> res_t =
        rotation_L_.transpose() *
        (R.transpose() * (rotation_R_ * (t - translation_R_) - t) -
         translation_L_);
    if (jacobian_matrix != nullptr) {
      *jacobian_matrix = JacobianMatrix::Zero();

      jacobian_matrix->row(0) = res_R(0, 0).v; 
      jacobian_matrix->row(1) = res_R(1, 0).v; 
      jacobian_matrix->row(2) = res_R(2, 0).v; 

      jacobian_matrix->row(3) = res_R(0, 1).v; 
      jacobian_matrix->row(4) = res_R(1, 1).v; 
      jacobian_matrix->row(5) = res_R(2, 1).v; 

      jacobian_matrix->row(6) = res_R(0, 2).v; 
      jacobian_matrix->row(7) = res_R(1, 2).v; 
      jacobian_matrix->row(8) = res_R(2, 2).v; 

      jacobian_matrix->row(9) = res_t(0).v;
      jacobian_matrix->row(10) = res_t(1).v;
      jacobian_matrix->row(11) = res_t(2).v;
    }
    return true;
  }

  double Evaluate(const Eigen::VectorXd &x) const override {

    ResidualVector v;
    v.setZero();
    Evaluate(x, &v, nullptr);
    
    return 0.5 * v.dot(v);
  }

  Eigen::VectorXd Jacobian(const Eigen::VectorXd &x) const override {
    ResidualVector v;
    v.setZero();
    JacobianMatrix jacobian;
    Evaluate(x, &v, &jacobian);
    return v.transpose() * jacobian;
  }

  Eigen::VectorXd ProjectExtendedGradientToTangentSpace(
      const Eigen::VectorXd &x,
      const Eigen::VectorXd &general_gradient) const override {
    return SepecialEuclideanManifold::Project(x, general_gradient);
  }

  Eigen::VectorXd Move(const Eigen::VectorXd &x,
                       const Eigen::VectorXd &direction) const override {
    return SepecialEuclideanManifold::Retraction(x, direction);
  }
};

TEST(LeastQuaresRiemannGredientDescentLinearSearch, SepecialEuclideanManifold) {
  Eigen::Matrix3d RA1, RA2;
  Eigen::Matrix3d RB1, RB2; 
  Eigen::Vector3d TA1, TA2, TB1, TB2;
  
  RA1 << -0.989992, -0.14112,  0.000,
         0.141120 , -0.989992, 0.000,
         0.000000 ,  0.00000, 1.000;
  TA1 << 0.0, 0.0, 0.0;

  RA2 << 0.07073, 0.000000, 0.997495, 
         0.000000, 1.000000, 0.000000,
         -0.997495, 0.000000, 0.070737;
  TA2 << -400, 0, 400;

  RB1 << -0.989992, -0.138307, 0.028036, 
         0.138307 , -0.911449, 0.387470, 
         -0.028036 ,  0.387470, 0.921456;

  TB1 << -26.9559,
         -96.1332,
          19.4872;

  RB2 <<  0.070737, 0.198172, 0.997612, 
         -0.198172, 0.963323, -0.180936,
         -0.977612, -0.180936, 0.107415;
  TB2 << -309.543, 59.0244, 291.177;

  std::shared_ptr<SpecialEuclideanManifoldCostFunction> cost_function = 
      std::make_shared<SpecialEuclideanManifoldCostFunction>(RA1, RB1, TA1,
                                                             TB1);
  using Manifold = Eigen::Matrix<double, 12, 1>;
  Eigen::VectorXd manifold = Manifold::Zero();
  rgd(cost_function, &manifold);
}

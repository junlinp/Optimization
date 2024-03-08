#include "gradient_checker.h"
#include "rgd.h"

#include "gtest/gtest.h"
#include <memory>
#include "manifold.h"
#include "rgd_cost_function_interface.h"
#include "so3_cost_function_interface.h"
#include "ceres/rotation.h"
#include "ceres/autodiff_cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"

#include "RGD/gradient_checker.h"

#include "ceres/jet.h"
struct BasicCostFunction : public RGDFirstOrderInterface {
   private:
    Eigen::Matrix3d target;

   public:
    BasicCostFunction() { target << 1, 0, 0, 0, 0, 1, 0, -1, 0; }

    double Evaluate(const Eigen::VectorXd &x) const override {

      Eigen::Map<const Eigen::Matrix3d> X(x.data());
      Eigen::Matrix3d H = target.transpose() * X - Eigen::Matrix3d::Identity();
      return 0.5 * (H.transpose() * H).trace();
    }

    Eigen::VectorXd Jacobian(
        const Eigen::VectorXd &x) const override {

      Eigen::Matrix3d A = target.eval();
      Eigen::Map<const Eigen::Matrix3d> X(x.data());
      Eigen::Matrix3d H =  A * (A.transpose() * X - Eigen::Matrix3d::Identity()); 
      
      Eigen::Map<Eigen::Matrix<double, 9, 1>> jacobian(H.data());
      return jacobian;
    }

    Eigen::VectorXd ProjectExtendedGradientToTangentSpace(
      const Eigen::VectorXd&x, const Eigen::VectorXd &general_gradient) const override {
        return RotationMatrixManifold::Project(x, general_gradient);
      };

    Eigen::VectorXd Move(const Eigen::VectorXd& x,
                                 const Eigen::VectorXd& direction) const override {
      return RotationMatrixManifold::Retraction(x, direction);
    }
  };


TEST(Gradientchecker, Basic) {
  std::shared_ptr<RGDFirstOrderInterface> function = std::make_shared<BasicCostFunction>();

  GradientChecker::Check<RotationMatrixManifold>(function);
}

TEST(RGD, Basic) {
  std::shared_ptr<BasicCostFunction> function = std::make_shared<BasicCostFunction>();
  Eigen::Matrix<double, 9, 1> x0;
  double theta = 3.14 * 0.4;
  x0 << 1.0, 0.0, 0.0, 0.0, std::cos(theta), -std::sin(theta), 0.0,
      std::sin(theta), std::cos(theta);
  Eigen::VectorXd x = x0;
  rgd(function, &x);
  Eigen::Map<Eigen::Matrix3d> matrix_solution(x.data());
  
  EXPECT_LT(function->Evaluate(x), 1e-5);
}

struct GradientCheckerExample : public RiemannianSecondOrderInterface {

    public:
      Eigen::Matrix3d symmetry_A;
    
    GradientCheckerExample(Eigen::Matrix3d A) : symmetry_A(A) {}

    double Evaluate(const Eigen::VectorXd &x) const override {
      return -x.dot(symmetry_A * x);
    }


    Eigen::VectorXd Jacobian(
        const Eigen::VectorXd &x) const override {
          return -2.0 * symmetry_A * x;
    }

    Eigen::VectorXd ProjectExtendedGradientToTangentSpace(
      const Eigen::VectorXd&x, const Eigen::VectorXd &general_gradient) const override {
        return SphereManifold<3>::Project(x, general_gradient);
      };

    Eigen::VectorXd Move(const Eigen::VectorXd& x,
                                 const Eigen::VectorXd& direction) const override {
      return SphereManifold<3>::Retraction(x, direction);
    }

    Eigen::MatrixXd Hess(const Eigen::VectorXd& x) const override {
      Eigen::VectorXd tmp = symmetry_A * x;
      int n = tmp.rows();

      Eigen::MatrixXd H(n, n);

      for (int i = 0; i < n; i++) {
        H.col(i) = x(i) * tmp;
      }
      return H;
    }
};

TEST(GradientCheckerExample, SphereManifold) {
  Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
  A << 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0;
  std::shared_ptr<RGDFirstOrderInterface> cost_function =
      std::make_shared<GradientCheckerExample>(A);

  GradientChecker::Check<SphereManifold<3>>(cost_function);
}


TEST(rgd, SphereManifold) {
  Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
  A << 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0;
  std::shared_ptr<RGDFirstOrderInterface> cost_function =
      std::make_shared<GradientCheckerExample>(A);

  Eigen::VectorXd x0 = SphereManifold<3>::IdentityElement();

  rgd(cost_function, &x0);
  EXPECT_LE(std::abs(x0.squaredNorm() - 1.0), 1e-5);
  EXPECT_LE(std::abs((-x0.dot(A * x0) + 3.0)), 1e-5);
}


TEST(Newton, SphereManifold) {
  Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
  A << 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0;
  std::shared_ptr<RiemannianSecondOrderInterface> cost_function =
      std::make_shared<GradientCheckerExample>(A);
  Eigen::VectorXd x0 = SphereManifold<3>::IdentityElement();
  RiemannianNewtonMethod(cost_function, &x0);
  Eigen::Vector3d tangent;
  EXPECT_LE(std::abs(x0.squaredNorm() - 1.0), 1e-5);
  EXPECT_LE(std::abs((-x0.dot(A * x0) + 3.0)), 1e-5);
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
      : rotation_L_(rotation_R),
        rotation_R_(rotation_L),
        translation_L_(translation_R),
        translation_R_(translation_L) {}

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
        rotation_L_.transpose() * R.transpose() * rotation_R_ * R - Eigen::Matrix3d::Identity();

    Eigen::Matrix<JetType, 3, 1> tmp = rotation_R_ * t + translation_R_;
    tmp = R.transpose() * (tmp - t);

    Eigen::Matrix<JetType, 3, 1> res_t =
        rotation_L_.transpose() * (tmp - translation_L_);

    (*residuals)(0) = res_R(0, 0).a;
    (*residuals)(1) = res_R(1, 0).a;
    (*residuals)(2) = res_R(2, 0).a;
    (*residuals)(3) = res_R(0, 1).a;
    (*residuals)(4) = res_R(1, 1).a;
    (*residuals)(5) = res_R(2, 1).a;
    (*residuals)(6) = res_R(0, 2).a;
    (*residuals)(7) = res_R(1, 2).a;
    (*residuals)(8) = res_R(2, 2).a;

    (*residuals)(9) = res_t(0).a;
    (*residuals)(10) = res_t(1).a;
    (*residuals)(11) = res_t(2).a;

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

class MultipleCostFunction : public RGDFirstOrderInterface {
public:
  MultipleCostFunction() = default;
  ~MultipleCostFunction() = default; 

  void AddCostFunction(std::shared_ptr<RGDFirstOrderInterface> ptr) {
    cost_function_ptr.push_back(ptr);
  }

  double Evaluate(const Eigen::VectorXd &x) const override {
    double res = 0.0;

    for(auto ptr : cost_function_ptr) {
      res += ptr->Evaluate(x);
    }
    return res;
  }

  Eigen::VectorXd Jacobian(const Eigen::VectorXd &x) const override {
    Eigen::VectorXd jacobian = x;
    jacobian.setZero();

    for (auto ptr : cost_function_ptr) {
      jacobian += ptr->Jacobian(x);
    }
    return jacobian;
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

private:
  std::vector<std::shared_ptr<RGDFirstOrderInterface>> cost_function_ptr;
};

TEST(RotationMatrixManifold, tangent_space) {
  Eigen::Matrix<double, 9, 1> manifold;
  manifold << 1.0, 0.0, 0.0, 0.0, 0.98014571, 0.19827854, 0.0, -0.19827854, 0.98014571;

  Eigen::Map<Eigen::Matrix3d> rotation_matrix(manifold.data());
  EXPECT_LT(((rotation_matrix.transpose() * rotation_matrix).transpose() - Eigen::Matrix3d::Identity()).array().square().sum(), 1e-5);

  Eigen::Matrix<double, 9, 1> tangent_space_example;

  tangent_space_example << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;

  Eigen::Matrix<double, 9, 1> tangent_space = RotationMatrixManifold::Project(manifold, tangent_space_example);

  Eigen::Matrix<double, 9, 1> next_manifold = RotationMatrixManifold::Retraction(manifold, tangent_space);

  Eigen::Map<Eigen::Matrix3d> next_rotation_matrix(next_manifold.data());

  EXPECT_LT(((next_rotation_matrix.transpose() * next_rotation_matrix).transpose() - Eigen::Matrix3d::Identity()).array().square().sum(), 1e-5);
  
}

TEST(SpecialEuclideanManifoldCostFunction, fval) {
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

 Eigen::Matrix3d target_rotation;
 target_rotation << 1.0, 0.0, 0.0, 0.0, 0.98014571, -0.19827854, 0.0, 0.19827854, 0.98014571;
 Eigen::Vector3d target_translation;
 target_translation << 9.99989033, 50.00001805, 99.9999990214;

 double rotation_error = (RB1 * target_rotation - target_rotation - RA1).array().square().sum();
 EXPECT_LE(rotation_error, 5);
 Eigen::Matrix<double, 12, 1> manifold;
 manifold << 1.0, 0.0, 0.0, 0.0, 0.98014571, 0.19827854, 0.0, -0.19827854, 0.98014571, 10.0, 50.0, 100.0;

 SpecialEuclideanManifoldCostFunction function(RA1, RB1, TA1, TB1);
 double error = function.Evaluate(manifold);
 
 EXPECT_LE(error ,rotation_error);
}

TEST(SpecialEuclideanManifoldCostFunction, gradient) {
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

 Eigen::Matrix3d target_rotation;
 target_rotation << 1.0, 0.0, 0.0, 0.0, 0.98014571, -0.19827854, 0.0, 0.19827854, 0.98014571;
 Eigen::Vector3d target_translation;
 target_translation << 9.99989033, 50.00001805, 99.9999990214;

  Eigen::Matrix3d target_translation_gradient = RB1.transpose() * target_rotation.transpose() * (RA1 - Eigen::Matrix3d::Identity());

 Eigen::Matrix<double, 12, 1> manifold;
 manifold << 1.0, 0.0, 0.0, 0.0, 0.98014571, 0.19827854, 0.0, -0.19827854, 0.98014571, 10.0, 50.0, 100.0;

 SpecialEuclideanManifoldCostFunction function(RB1, RA1, TB1, TA1);

  using ResidualVector = Eigen::Matrix<double, 9 + 3, 1>;
  using JacobianMatrix = Eigen::Matrix<double, 9 + 3, 9 + 3>;
 ResidualVector res;

 JacobianMatrix J;

 function.Evaluate(manifold, & res, &J);

 Eigen::Matrix3d function_translation_gradient = J.block(9,9, 3, 3);

 double diff = (function_translation_gradient - target_translation_gradient).array().square().sum(); 
 std::cout << J << std::endl;
 EXPECT_LE(diff, 1e-6);
}

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

  std::shared_ptr<SpecialEuclideanManifoldCostFunction> cost_function1 = 
      std::make_shared<SpecialEuclideanManifoldCostFunction>(RA1, RB1, TA1,
                                                             TB1);

  std::shared_ptr<SpecialEuclideanManifoldCostFunction> cost_function2 = 
      std::make_shared<SpecialEuclideanManifoldCostFunction>(RA2, RB2, TA2,
                                                             TB2);
  
  std::shared_ptr<MultipleCostFunction> cost_function = std::make_shared<MultipleCostFunction>();
  cost_function->AddCostFunction(cost_function1);
  cost_function->AddCostFunction(cost_function2);

  //using Manifold = Eigen::Matrix<double, 12, 1>;
  Eigen::VectorXd manifold = ProductManifold<RotationMatrixManifold, EuclideanManifold<3>>::IdentityElement();

  //manifold << 1.0, 0.0, 0.0, 0.0, 0.98014571, 0.19827854, 0.0, -0.19827854, 0.98014571, 10.0, 50.0, 100.0;
  manifold << 1.0, 0.0, 0.0, 0.0, 0.98014571, 0.0, 0.0, 0.0, 0.98014571, 10.0, 50.0, 50.0;

  rgd(cost_function, &manifold);
  //[[ 1.          0.          0.          9.99989033]
 //[ 0.          0.98014571 -0.19827854 50.00001805]
 //[ 0.          0.19827854  0.98014571 99.99990214]
 //[ 0.          0.          0.          1.        ]]

 Eigen::Matrix3d target_rotation;
 target_rotation << 1.0, 0.0, 0.0, 0.0, 0.98014571, -0.19827854, 0.0, 0.19827854, 0.98014571;
 Eigen::Vector3d target_translation;
 target_translation << 9.99989033, 50.00001805, 99.9999990214;

 Eigen::Vector3d solution_translation = manifold.tail<3>();

 std::cout << "target_rotation : " << target_rotation << std::endl;
 std::cout << "Solution_translation : " << solution_translation << std::endl;

 EXPECT_LE((target_translation - solution_translation).norm(), 1.0);
 Eigen::Map<Eigen::Matrix3d> solution_rotation(manifold.data());

 std::cout << "solution_rotation : " << solution_rotation << std::endl;

 double rotation_error = (target_rotation.transpose() * solution_rotation -
                          Eigen::Matrix3d::Identity())
                             .array()
                             .square()
                             .sum() *
                         0.5;
 EXPECT_LE(rotation_error, 0.5);
}

struct ConjugationCostFunction{
  Eigen::Matrix3d rotation_A_, rotation_B_;
  Eigen::Vector3d translation_A_, translation_B_;

 public:
  ConjugationCostFunction(const Eigen::Matrix3d &rotation_A,
               const Eigen::Matrix3d &rotation_B,
               const Eigen::Vector3d &translation_A,
               const Eigen::Vector3d &translation_B)
      : rotation_A_(rotation_A),
        rotation_B_(rotation_B),
        translation_A_(translation_A),
        translation_B_(translation_B) {}

  template <typename T>
  bool operator()(const T *const parameters, T *residuals) const {
    using TMatrix3d = Eigen::Matrix<T, 3, 3>;
    using TVector3d = Eigen::Matrix<T, 3, 1>;

    TMatrix3d rotation;
    ceres::AngleAxisToRotationMatrix(parameters, rotation.data());
    TVector3d translation;
    translation << parameters[3], parameters[4], parameters[5];
    // solve AX = XB
    // => B' * X' * A * X = Identity

    TMatrix3d res_rotation =
        (rotation_B_.transpose() * rotation.transpose() * rotation_A_ * rotation) - Eigen::Matrix3d::Identity();

    TVector3d tmp = rotation_A_ * translation + translation_A_;
    tmp = rotation.transpose() * (tmp - translation);
    tmp = rotation_B_.transpose() * (tmp - translation_B_);
    TVector3d res_translation = tmp;

    Eigen::Map<Eigen::Matrix<T, 3, 3>> residuals_map(residuals);
    residuals_map = 2.0 * res_rotation;
    residuals[9] = res_translation(0);
    residuals[10] = res_translation(1);
    residuals[11] = res_translation(2);
    return true;
  }
};

TEST(ceres, ConjugationRtationAveraging) {

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



  ceres::CostFunction *cost_function =
      new ceres::AutoDiffCostFunction<ConjugationCostFunction, 12, 6>(
          new ConjugationCostFunction(RA1, RB1, TA1, TB1));

  ceres::CostFunction *cost_function2 =
      new ceres::AutoDiffCostFunction<ConjugationCostFunction, 12, 6>(
          new ConjugationCostFunction(RA2, RB2, TA2, TB2));

  ceres::Problem problem;
  double parameters[6] = {1.0, 2.0, 3.0, 9.0, 0.0, 0.0};
  problem.AddResidualBlock(cost_function, nullptr, parameters);
  problem.AddResidualBlock(cost_function2, nullptr, parameters);

  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  options.max_num_iterations = 1024;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << std::endl;

  Eigen::Matrix3d rotation;
  ceres::AngleAxisToRotationMatrix(parameters, rotation.data());

  Eigen::Matrix3d target_rotation;
  target_rotation << 1.0, 0.0, 0.0, 0.0, 0.98014571, -0.19827854, 0.0,
      0.19827854, 0.98014571;
  Eigen::Vector3d target_translation;
  target_translation << 9.99989033, 50.00001805, 99.9999990214;

  std::cout << rotation << std::endl;
  std::cout << target_rotation << std::endl;
  Eigen::Vector3d translation;
  translation << parameters[3], parameters[4], parameters[5];

  std::cout << translation << std::endl;
  std::cout << target_translation << std::endl;

  std::cout << RB1 * target_rotation << std::endl;
  std::cout << target_rotation * RA1 << std::endl;
}

#include "gtest/gtest.h"
#include "lm.h"

class QuadraticFunction : public BinaryFunction {
 public:
  QuadraticFunction(size_t n): dimension_(n) {
    A_ = Eigen::MatrixXd::Random(n, n);
    A_ = A_.transpose() * A_;
    b_ = Eigen::VectorXd::Random(n);
  }

  size_t VariableDimension() const override { return dimension_; }
  Eigen::MatrixXd Jacobians(const Eigen::VectorXd& x) const override {
    return A_;
  }

  Eigen::VectorXd Evaluate(const Eigen::VectorXd& x) const override {
    return A_ * x + b_;
  }

  void BinaryJacobians(Eigen::MatrixXd* first, Eigen::MatrixXd* second) const override {
    *first =  A_.block(0, 0, dimension_, dimension_ / 2);
    *second = A_.block(0, dimension_ / 2, dimension_, dimension_ - dimension_ / 2);
  }

 private:
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
  size_t dimension_;
};

//TEST(LMSolver, Basic) {
    //LMSolver solver;
    //std::shared_ptr<FirstOrderFunction> function = std::make_shared<QuadraticFunction>(1024);
    //solver.SetFunction(function);
    //Eigen::VectorXd x = Eigen::VectorXd::Zero(function->VariableDimension());
    //std::cout << "Solve" << std::endl;
    //solver.Solve(&x);

    //Eigen::VectorXd residual = function->Evaluate(x);
    //EXPECT_LT(0.5 * residual.squaredNorm(), 1e-6);
//}

TEST(BundleAdjustmentLMSolver, Basic) {
    BundleAdjustmentLMSolver solver;
    std::shared_ptr<BinaryFunction> function = std::make_shared<QuadraticFunction>(1024);
    solver.SetFunction(function);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(function->VariableDimension());
    //solver.Solve(&x);
    //Eigen::VectorXd residual = function->Evaluate(x);
    //EXPECT_LT(0.5 * residual.squaredNorm(), 1e-6);
}

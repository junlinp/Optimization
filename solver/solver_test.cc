#include "gtest/gtest.h"
#include "lm.h"

class QuadraticFunction : public FirstOrderFunction {
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

 private:
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
  size_t dimension_;
};
TEST(LMSolver, Basic) {
    LMSolver solver;
    std::shared_ptr<FirstOrderFunction> function = std::make_shared<QuadraticFunction>(1024 * 4);
    solver.SetFunction(function);

    Eigen::VectorXd x = Eigen::VectorXd::Zero(function->VariableDimension());

    solver.Solve(&x);

    Eigen::VectorXd residual = function->Evaluate(x);
    EXPECT_LT(0.5 * residual.squaredNorm(), 1e-6);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
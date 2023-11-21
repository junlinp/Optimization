#include "gtest/gtest.h"
#include <random>
#include "first_oracle.h"
#include "gradient_descent.h"

class QuadraticFunction : public FirstOracle {
    Eigen::MatrixXd A_;
    Eigen::VectorXd b_;
    public:
    QuadraticFunction(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) : A_(A), b_(b) {}

    double fval(const Eigen::VectorXd& x) override {
      return 0.5 * x.dot(A_ * x) - b_.dot(x);
    }

    Eigen::VectorXd SubGradient(const Eigen::VectorXd &x) override {
        return A_ * x - b_;
    }
    ~QuadraticFunction() = default;
};

class FirstOrderTest : public testing::Test {
    public:
    void SetUp() override {
        L_ = 1024 * 2;
        unif_dis = std::uniform_real_distribution<double>(0, L_);
        int n = 2048 * 4;
        Eigen::VectorXd diag(n);
        for (int i = 0; i < n; i++) {
            diag(i) = unif_dis(seeds);
        }
        Eigen::MatrixXd A = diag.asDiagonal();
        Eigen::VectorXd b = Eigen::VectorXd::Random(n);
        function_ = std::make_shared<QuadraticFunction>(A, b);
        x0_ = Eigen::VectorXd::Zero(n);
    }

    public:
      std::shared_ptr<FirstOracle> function_;
      Eigen::VectorXd x0_;
      double L_;
      std::mt19937 seeds;
      std::uniform_real_distribution<double> unif_dis;
};

TEST_F(FirstOrderTest, gradient_decent_with_L) {
    Eigen::VectorXd x_opt = GradientDescnet(function_, x0_,  L_);

    std::cout << "Initial fval : " << function_->fval(x0_) << std::endl;
    std::cout << "Final fval : " << function_->fval(x_opt) << std::endl;
}

TEST_F(FirstOrderTest, gradient_decent) {
    Eigen::VectorXd x_opt = GradientDescnet(function_, x0_);
    std::cout << "Initial fval : " << function_->fval(x0_) << std::endl;
    std::cout << "Final fval : " << function_->fval(x_opt) << std::endl;
}

TEST_F(FirstOrderTest, nesterov_gradient_decent_with_L) {
    Eigen::VectorXd x_opt = NesterovGradientDescnet(function_, x0_,  L_);
    std::cout << "Initial fval : " << function_->fval(x0_) << std::endl;
    std::cout << "Final fval : " << function_->fval(x_opt) << std::endl;
}

TEST_F(FirstOrderTest, nesterov_gradient_decent) {
    Eigen::VectorXd x_opt = NesterovGradientDescnet(function_, x0_);
    std::cout << "Initial fval : " << function_->fval(x0_) << std::endl;
    std::cout << "Final fval : " << function_->fval(x_opt) << std::endl;
}

TEST_F(FirstOrderTest, nesterov_gradient_decent2) {
    Eigen::VectorXd x_opt = NesterovGradientDescnet2(function_, x0_,  L_);
    std::cout << "Initial fval : " << function_->fval(x0_) << std::endl;
    std::cout << "Final fval : " << function_->fval(x_opt) << std::endl;
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
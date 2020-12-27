#include "gtest/gtest.h"
#include "Eigen/Dense"
#include "linear_solver.h"
#include "linear_programing.h"
TEST(Conjugate_Gradient, PSD) {
    int n = 1024;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n); 
    Eigen::VectorXd b = Eigen::VectorXd::Random(n);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);

    A = A * A.transpose(); 
    ConjugateGradient(A, b, x);

    std::cout << "Conjugate Gradient Error : " << (A*x - b).norm() << std::endl;
}

TEST(LP, Test_Case) {
    Eigen::VectorXd c(4);
    c << 1, -1, 0, 0;
    Eigen::MatrixXd A(2, 4);
    A << 10, -7, -1, 0,
         1, 0.5, 0, 1;
    Eigen::VectorXd b(2);
    b << 5.0, 3.0;
    Eigen::VectorXd x;
    LPSolver(c, A, b, x);
    std::cout << "x : " << x << std::endl;
    std::cout << "A * x - b : " << b - A*x << std::endl;
}
int main(int argc, char** argv) {
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
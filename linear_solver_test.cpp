#include "gtest/gtest.h"
#include "Eigen/Dense"
#include "linear_solver.h"
TEST(Conjugate_Gradient, PSD) {
    int n = 1024;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n); 
    Eigen::VectorXd b = Eigen::VectorXd::Random(n);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);

    A = A * A.transpose(); 
    ConjugateGradient(A, b, x);

    std::cout << "Conjugate Gradient Error : " << (A*x - b).norm() << std::endl;
}
int main(int argc, char** argv) {
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
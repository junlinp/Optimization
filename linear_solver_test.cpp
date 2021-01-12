#include "gtest/gtest.h"
#include "Eigen/Dense"
#include "linear_solver.h"
#include "linear_programing.h"
/*
TEST(Conjugate_Gradient, PSD) {
    int n = 1024;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n); 
    Eigen::VectorXd b = Eigen::VectorXd::Random(n);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);

    A = A * A.transpose(); 
    ConjugateGradient(A, b, x);

    std::cout << "Conjugate Gradient Error : " << (A*x - b).norm() << std::endl;
}
*/
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

TEST(LP, Test_Case2) {
    Eigen::VectorXd c(3);
    c << 2.0, 2.0, 0.0;
    Eigen::MatrixXd A(1, 3);
    A << 1.0, 1.0, -1.0;
    Eigen::VectorXd b(1);
    b << 3;
    Eigen::VectorXd x;
    LPSolver(c, A, b, x);
    // Should be (1.5, 1.5, 0.0)
    std::cout << "x : " << x << std::endl;
    // Should be 6.0
    std::cout << "Optimal Value : " << c.dot(x) << std::endl;
}
TEST(LP, Test_Case3) {
    Eigen::VectorXd c(5);
    c << -3.0, -5.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd A(3, 5);
    A << 1.0, 0.0, 1.0, 0.0, 0.0,
         0.0, 2.0, 0.0, 1.0, 0.0,
         3.0, 2.0, 0.0, 0.0, 1.0;
    Eigen::VectorXd b(3);
    b << 3,
    12,
    18;
    Eigen::VectorXd x;
    LPSolver(c, A, b, x);
    // Should be (2.0, 6.0, 0.0, 0.0, 0.0)
    std::cout << "x : " << x << std::endl;
    // Should be -36.0
    std::cout << "Optimal Value : " << c.dot(x) << std::endl;
}
TEST(SDP, Test_Case) {
    Eigen::MatrixXd C(7, 7);
    C.setZero();
    C(0, 2) = C(2, 0)= 0.5;

    Eigen::MatrixXd A1(7, 7), A2(7, 7), A3(7, 7) ,A4(7, 7);
    A1.setZero();
    A1(0, 1) = A1(1, 0) = 0.5;
    A1(3, 3) = 1.0;
    A2.setZero();
    A2(0, 1) = A2(1, 0)= 0.5;
    A2(4, 4) = -1.0;
    A3.setZero();
    A3(1, 2) = A3(2, 1) = 0.5;
    A3(5, 5) = 1.0;
    A4.setZero();
    A4(1, 2) = A4(2, 1)= 0.5;
    A4(6, 6) = -1.0;

    Eigen::MatrixXd A5(7, 7), A6(7, 7), A7(7, 7) ;
    A5.setZero();
    A5(0, 0) = 1.0;

    A6.setZero();
    A6(1, 1) = 1.0;
    
    A7.setZero();
    A7(2, 2) = 1.0;
    //std::cout << "A7 : "<< A7 << std::endl;
    std::vector<Eigen::MatrixXd> A;
    A.push_back(A1);
    A.push_back(A2);
    A.push_back(A3);
    A.push_back(A4);
    A.push_back(A5);
    A.push_back(A6);
    A.push_back(A7);

    Eigen::VectorXd b(7);
    b << -0.1, -0.2, 0.5, 0.4, 1.0, 1.0, 1.0;
    Eigen::MatrixXd x;
    SymmetricSolver(C, A, b, x);

    // optimal value -0.978
    std::cout << "tr(C*X) : " << (C * x).trace() << std::endl;
    std::cout << "X : " << x << std::endl;
    std::cout << "det(X) : " << x.determinant() << std::endl;
}
int main(int argc, char** argv) {
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
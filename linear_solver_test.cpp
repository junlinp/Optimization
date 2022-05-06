#include "gtest/gtest.h"
#include "Eigen/Dense"
#include "linear_solver.h"
#include "linear_programing.h"
#include <fstream>
#include "LinearProgramingConfig.h"
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


TEST(LP2, Test_Case) {
    Eigen::VectorXd c(4);
    c << 1, -1, 0, 0;
    Eigen::MatrixXd A(2, 4);
    A << 10, -7, -1, 0,
         1, 0.5, 0, 1;
    Eigen::VectorXd b(2);
    b << 5.0, 3.0;
    Eigen::VectorXd x(4);
    LPSolver2(c, A, b, x);
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
    Eigen::VectorXd x(3);

    LPSolver(c, A, b, x);
    // Should be (1.5, 1.5, 0.0)
    std::cout << "x : " << x << std::endl;
    // Should be 6.0
    std::cout << "Optimal Value : " << c.dot(x) << std::endl;
}

TEST(LP2, Test_Case2) {
    Eigen::VectorXd c(3);
    c << 2.0, 2.0, 0.0;
    Eigen::MatrixXd A(1, 3);
    A << 1.0, 1.0, -1.0;
    Eigen::VectorXd b(1);
    b << 3;
    Eigen::VectorXd x(3);
    LPSolver2(c, A, b, x);
    // X Should be (1.5, 1.5, 0.0)
    // Optimized Value Should be 6.0
    EXPECT_NEAR(x(0), 1.5, 1e-6);
    EXPECT_NEAR(x(1), 1.5, 1e-6);
    EXPECT_NEAR(x(2), 0.0, 1e-6);
    EXPECT_NEAR(c.dot(x), 6.0, 1e-6);
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
    // Should be (2.0, 6.0, 1.0, 0.0, 0.0)
    // Should be -36.0
    double EPSILON = 1e-7;
    EXPECT_NEAR(x(0), 2.0, EPSILON);
    EXPECT_NEAR(x(1), 6.0, EPSILON);
    EXPECT_NEAR(x(2), 1.0, EPSILON);
    EXPECT_NEAR(x(3), 0.0, EPSILON);
    EXPECT_NEAR(x(4), 0.0, EPSILON);
    EXPECT_NEAR(c.dot(x), -36.0, EPSILON);
}

TEST(LP2, Test_Case3) {
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
    Eigen::VectorXd x(5);
    LPSolver2(c, A, b, x);
    // Should be (2.0, 6.0, 1.0, 0.0, 0.0)
    // Should be -36.0
    double EPSILON = 1e-7;
    EXPECT_NEAR(x(0), 2.0, EPSILON);
    EXPECT_NEAR(x(1), 6.0, EPSILON);
    EXPECT_NEAR(x(2), 1.0, EPSILON);
    EXPECT_NEAR(x(3), 0.0, EPSILON);
    EXPECT_NEAR(x(4), 0.0, EPSILON);
    EXPECT_NEAR(c.dot(x), -36.0, EPSILON);
}
TEST(SOCP, Test_Case1) {

}


auto Hamming_Load_Data(const std::string& data_set) {
    std::string direct = HAMMING_DATASET_PATH + std::string("/") + data_set;
    std::vector<double> c_coeff;
    std::ifstream ifs(direct + "/c.txt");
    size_t index;
    double value;
    while(ifs >> index >> value) {
        c_coeff.push_back(value);
    }
    ifs.close();
    size_t n = c_coeff.size();
    Eigen::VectorXd c(n);
    for(int i = 0; i < n; i++) {
        c(i) = c_coeff[i];
    }


    std::vector<double> b_coeff;
    ifs.open(direct + "/b.txt");
    while(ifs >> index >> value) {
        b_coeff.push_back(value);
    }
    size_t m = b_coeff.size();
    Eigen::VectorXd b(m);
    for(int i = 0; i < m; i++) {
        b(i) = b_coeff[i];
    }
    ifs.close();
    Eigen::MatrixXd A(m, n);
    A.setZero();
    ifs.open(direct + "/A_sparse.txt");
    size_t row, col;
    while(ifs >> row >> col >> value) {
        A(row, col) = value;
    }
    ifs.close();

    return std::tuple<Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd>(c, A, b);
}

TEST(SDP, HAMMING_7_5_6) {

    auto [c, A, b] = Hamming_Load_Data("7_5_6");

    // |V| = 128
    // |E| = 1793
    // Optimizaed Value : 422/3
    Eigen::MatrixXd X(128, 128);
    SDPSolver(c, A, b, X);

    EXPECT_NEAR(c.dot(SemiDefineSpace::Vec(X)) * 3, 422.0, 1e-5);
}

TEST(SDP, HAMMING_8_3_4) {
    // |V| = 256
    // |E| = 16129
    // Optimizaed Value : 25.6
    auto [c, A, b] = Hamming_Load_Data("8_3_4");

    size_t vertex = 256;
    Eigen::MatrixXd X(vertex, vertex);
    SDPSolver(c, A, b, X);
    EXPECT_NEAR(c.dot(SemiDefineSpace::Vec(X)), 25.6, 1e-5);
}
TEST(SDP, HAMMING_9_8) {
    // |V| = 512
    // |E| = 2305
    // Optimizaed Value : 224

    auto [c, A, b] = Hamming_Load_Data("9_8");
    size_t vertex = 512;
    Eigen::MatrixXd X(vertex, vertex);
    SDPSolver(c, A, b, X);
    EXPECT_NEAR(c.dot(SemiDefineSpace::Vec(X)), 224, 1e-5);
}
TEST(SDP, HAMMING_9_5_6) {
    // |V| = 512
    // |E| = 53761
    // Optimizaed Value : 851/3
    auto [c, A, b] = Hamming_Load_Data("9_5_6");
    size_t vertex = 512;
    Eigen::MatrixXd X(vertex, vertex);
    SDPSolver(c, A, b, X);
    EXPECT_NEAR(c.dot(SemiDefineSpace::Vec(X)) * 3, 851, 1e-5);
}
TEST(SDP, HAMMING_10_2) {

    // |V| = 1024
    // |E| = 23041
    // Optimizaed Value : 102.4
    auto [c, A, b] = Hamming_Load_Data("10_2");
    size_t vertex = 1024;
    Eigen::MatrixXd X(vertex, vertex);
    SDPSolver(c, A, b, X);
    EXPECT_NEAR(c.dot(SemiDefineSpace::Vec(X)), 102.4, 1e-5);
}
TEST(SDP, HAMMING_11_2) {
    // |V| = 2048
    // |E| = 56321
    // Optimizaed Value : 1702 / 3
    auto [c, A, b] = Hamming_Load_Data("11_2");
    size_t vertex = 2048;
    Eigen::MatrixXd X(vertex, vertex);
    SDPSolver(c, A, b, X);
    EXPECT_NEAR(c.dot(SemiDefineSpace::Vec(X)) * 3, 1702, 1e-5);
}
/*
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
*/
int main(int argc, char** argv) {
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
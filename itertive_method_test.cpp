#include "gtest/gtest.h"
#include "Eigen/Dense"
#include "chrono"
#include "lu.h"
#include "iostream"
TEST(Test, t) {
    EXPECT_TRUE(true);

}

TEST(HouseHolder, two_x_two) {
    Eigen::MatrixXd X(2, 2);
    X << -0.99984, 0.511211,
        -0.736924, -0.0826997;
    Eigen::MatrixXd Q(2, 2), R(2, 2);
    HouseHolder(X, Q, R);
    std::cout << "X : " << X <<std::endl;
    std::cout << "Q : " << Q <<std::endl;
    std::cout << "R : " << R <<std::endl;
    std::cout << "Q * Q: " << Q * Q.transpose() << std::endl;
}

TEST(HouseHolder, four_x_four) {
    const int n = 4;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, n);
    Eigen::MatrixXd Q(n, n), R(n, n);
    HouseHolder(X, Q, R);
    std::cout << "X : " << X <<std::endl;
    std::cout << "Q : " << Q <<std::endl;
    std::cout << "R : " << R <<std::endl;
    std::cout << "Q * Q: " << Q * Q.transpose() << std::endl;
}
TEST(HouseHolder, eight_x_eight) {
    const int n = 8;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, n);
    Eigen::MatrixXd Q(n, n), R(n, n);
    HouseHolder(X, Q, R);
    std::cout << "X : " << X <<std::endl;
    std::cout << "Q : " << Q <<std::endl;
    std::cout << "R : " << R <<std::endl;
    std::cout << "Q * Q: " << Q * Q.transpose() << std::endl;
}
int main() {
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}

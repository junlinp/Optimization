#include "conjugate_gradient.h"
#include "gtest/gtest.h"

TEST(ConjugateGradientTest, SimpleCase) {
    // Define a simple positive definite matrix A and vector b
    Eigen::MatrixXd A(2, 2);
    A << 4, 1,
         1, 3;
    Eigen::VectorXd b(2);
    b << 1,
         2;

    // Set a tolerance for the convergence
    double tolerance = 1e-6;

    // Call the ConjugateGradient function
    Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());
    ConjugateGradient(A, b, tolerance, & x);

    // Check if the solution is close to the expected result
    Eigen::VectorXd expected_solution(2);
    expected_solution << 1.0 / 11.0, 7.0 / 11.0; // Approximate expected solution
    EXPECT_TRUE((x - expected_solution).norm() < tolerance);
    std::cout << "Computed solution: " << x.transpose() << std::endl;
}

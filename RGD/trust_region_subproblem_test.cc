#include "trust_region_subproblem.h"
#include "gtest/gtest.h"

TEST(TrustRegionSubproblemTest, ExampleCase) {
    // Define a simple quadratic cost function
    const size_t dimension = 3;
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(dimension, dimension);
    Eigen::VectorXd B = -2 * Eigen::VectorXd::Ones(dimension);

    // Initial guess
    Eigen::VectorXd x_init(dimension);
    x_init.setRandom();

    // Define the trust region radius
    double trust_region_radius = 2.0;

    // Perform the trust region subproblem optimization
    TrustRegionSubProblem(A,B,trust_region_radius, &x_init);

    // Check if the new point is within the trust region
    EXPECT_LE((x_init).norm(), trust_region_radius);

    std::cout << (x_init) << std::endl;
    // Check if the cost is reduced
    // EXPECT_LT(cost_function(x_new), cost_function(x_init));

    
}

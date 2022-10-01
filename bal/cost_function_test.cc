#include "ceres/gradient_checker.h"
#include "gtest/gtest.h"

#include "cost_function.h"

TEST(ProjectTransform, GradientCheck) {

  ceres::CostFunction *my_cost_function = new ProjectTransformCostFunction();

  std::vector<double> parameter1 = {1.0, 2.0, 3.0};

  std::vector<double *> parameter_blocks;
  parameter_blocks.push_back(parameter1.data());
  ceres::NumericDiffOptions numeric_diff_options;
  ceres::GradientChecker gradient_checker(my_cost_function, nullptr,
                                          numeric_diff_options);
  ceres::GradientChecker::ProbeResults results;

  EXPECT_TRUE(gradient_checker.Probe(parameter_blocks.data(), 1e-9, &results));
}

TEST(RigidTransform, GradientCheck) {
    // TODO (junlinp)
}

TEST(RigidProjectTransform, GradientCheck) {
   // TODO (junlinp) 
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
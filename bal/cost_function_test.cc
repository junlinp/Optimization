#include "ceres/gradient_checker.h"
#include "ceres/manifold.h"
#include "ceres/rotation.h"
#include "ceres/autodiff_cost_function.h"


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
struct NormalCostFunction : public ceres::SizedCostFunction<4, 4>{
    public:

    bool Evaluate(double const* const* parameters, double* residual, double** jacobians) const override{
        const double* q = parameters[0];
        double scale = double(1.0) / sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);

        residual[0] = scale * q[0];
        residual[1] = scale * q[1];
        residual[2] = scale * q[2];
        residual[3] = scale * q[3];

        if (jacobians) {
            double* jacobian = jacobians[0];
            Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> map_jacobian(jacobian);
            Eigen::Map<const Eigen::Vector4d> map_quaternion(q);

            map_jacobian = 1.0 / scale * Eigen::Matrix4d::Identity() - 1.0 / scale /scale / scale * map_quaternion * map_quaternion.transpose();
        }
        return true;
    }

    template<class T>
    bool operator()(const T* quaternion, T*residual) const {
        const T* q = quaternion;
        T scale = T(1.0) / sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);

        residual[0] = scale * q[0];
        residual[1] = scale * q[1];
        residual[2] = scale * q[2];
        residual[3] = scale * q[3];
        return true;
    }
};

TEST(Normal, GradientCheck) {

    //ceres::CostFunction * cost_function = new ceres::AutoDiffCostFunction<NormalCostFunction, 4, 4>(new NormalCostFunction);
    ceres::CostFunction * cost_function = new NormalCostFunction();

    std::vector<double> quaternion = {0.5, 0.5, 0.0, 0.0};

    std::vector<double*> parameter_blocks;
    parameter_blocks.push_back(quaternion.data());

    ceres::NumericDiffOptions numeric_diff_options;
    ceres::GradientChecker gradient_checker(cost_function, nullptr, numeric_diff_options);
    ceres::GradientChecker::ProbeResults results;
    bool check_result = (gradient_checker.Probe(parameter_blocks.data(), 1e-9, &results));
    if (!check_result) {
        std::cout << results.error_log << std::endl;
    }
        std::cout << "Jacobian[0] : " << results.jacobians[0] << std::endl;
        std::cout << "local Jacobian[0] : " << results.local_jacobians[0] << std::endl;
        std::cout << "numerical Jacobian[0] : " << results.numeric_jacobians[0] << std::endl;
        std::cout << "numerical local jacobian[0] : " << results.local_numeric_jacobians[0] << std::endl;
        std::cout << "Result Vector : " << results.residuals << std::endl;

    EXPECT_TRUE(check_result);
    EXPECT_TRUE(results.return_value);
}
struct SE3CostFunction {
    public:
    template<class T>
    bool operator()(const T* quaternion, const T* point, T*residual) const {
        ceres::QuaternionRotatePoint(quaternion, point, residual);
        return true;
    }
};

TEST(SE3Transform, GradientCheck) {
    ceres::CostFunction * cost_function = new ceres::AutoDiffCostFunction<SE3CostFunction, 3, 4, 3>(new SE3CostFunction);

    std::vector<double> quaternion = {0.5, 0.5, 0.0, 0.0};
    std::vector<double> point = {3.0, 5.0, 7.0};

    std::vector<double*> parameter_blocks;
    parameter_blocks.push_back(quaternion.data());
    parameter_blocks.push_back(point.data());

    ceres::NumericDiffOptions numeric_diff_options;
    ceres::GradientChecker gradient_checker(cost_function, nullptr, numeric_diff_options);
    ceres::GradientChecker::ProbeResults results;
    bool check_result = (gradient_checker.Probe(parameter_blocks.data(), 1e-9, &results));
    if (!check_result) {
        std::cout << results.error_log << std::endl;
    }
        std::cout << "Jacobian[0] : " << results.jacobians[0] << std::endl;
        std::cout << "local Jacobian[0] : " << results.local_jacobians[0] << std::endl;
        std::cout << "numerical Jacobian[0] : " << results.numeric_jacobians[0] << std::endl;
        std::cout << "numerical local jacobian[0] : " << results.local_numeric_jacobians[0] << std::endl;
        std::cout << "Result Vector : " << results.residuals << std::endl;

    EXPECT_TRUE(check_result);
    EXPECT_TRUE(results.return_value);
}

TEST(RigidTransform, GradientCheck) {
    ceres::CostFunction * cost_function = new RigidTransformCostFunction();

    std::vector<double> quaternion = {0.5, 0.5, 0.0, 0.0};
    ceres::Manifold* quaternion_manifold = new ceres::QuaternionManifold();
    std::vector<double> transform = {-8.0, -10.0, -12.0};
    std::vector<double> point = {3.0, 5.0, 7.0};

    std::vector<double*> parameter_blocks;
    parameter_blocks.push_back(quaternion.data());
    parameter_blocks.push_back(transform.data());
    parameter_blocks.push_back(point.data());

    std::vector<const ceres::Manifold*> manifolds;
    manifolds.push_back(quaternion_manifold);
    manifolds.push_back(nullptr);
    manifolds.push_back(nullptr);

    ceres::NumericDiffOptions numeric_diff_options;
    ceres::GradientChecker gradient_checker(cost_function, nullptr, numeric_diff_options);
    ceres::GradientChecker::ProbeResults results;
    bool check_result = (gradient_checker.Probe(parameter_blocks.data(), 1e-9, &results));
    if (!check_result) {
        std::cout << results.error_log << std::endl;
        std::cout << "Jacobian[0] : " << results.jacobians[0] << std::endl;
        std::cout << "local Jacobian[0] : " << results.local_jacobians[0] << std::endl;
        std::cout << "numerical Jacobian[0] : " << results.numeric_jacobians[0] << std::endl;
        std::cout << "numerical local jacobian[0] : " << results.local_numeric_jacobians[0] << std::endl;
        std::cout << "Result Vector : " << results.residuals << std::endl;
    }

    EXPECT_TRUE(check_result);
    EXPECT_TRUE(results.return_value);

}

TEST(RigidProjectTransform, GradientCheck) {
   // TODO (junlinp) 
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
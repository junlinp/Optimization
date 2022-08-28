#include "ceres_bal_solver.h"

#include "ceres/autodiff_cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"

#include "evaluate.h"
#include "problem.h"

void CeresProblemSolver::Solve(Problem &problem) {
    ceres::Problem pro;
    for (auto&& [pairs, observation] : problem.observations_) {
        ceres::CostFunction* cost_func = new ceres::AutoDiffCostFunction<ProjectFunction, 2, 9, 3>(new ProjectFunction{observation.u(), observation.v()});

        CameraParam& camera_parameter = problem.cameras_[pairs.first];
        Landmark& points = problem.points_[pairs.second];

        pro.AddResidualBlock(cost_func, nullptr, camera_parameter.data(), points.data());
    }

    ceres::Solver::Options solver_options;
    solver_options.num_threads = 2;
    solver_options.minimizer_progress_to_stdout = true;
    solver_options.max_num_iterations = 500;
    ceres::Solver::Summary summary;

    ceres::Solve(solver_options, &pro, & summary);

    std::cout << summary.BriefReport() << std::endl;
    std::cout << "MSE : " << std::sqrt(summary.final_cost / problem.observations_.size()) << std::endl;
}
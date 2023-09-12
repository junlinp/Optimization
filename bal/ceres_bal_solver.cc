#include "ceres_bal_solver.h"

#include "ceres/problem.h"
#include "ceres/solver.h"

#include "problem.h"
#include <ceres/autodiff_cost_function.h>
#include <ceres/cost_function.h>
#include <ceres/types.h>

#include "cost_function_auto.h"

// camera model see
// http://grail.cs.washington.edu/projects/bal/
//
void CeresProblemSolver::Solve(Problem &problem) {
  ceres::Problem pro;
  for (auto &&[pairs, observation] : problem.observations_) {
    CameraParam &camera_parameter = problem.cameras_[pairs.first];
    Landmark &points = problem.points_[pairs.second];
    ceres::CostFunction *cost_func = ProjectFunction::CreateCostFunction(
        observation.u(), observation.v());
    pro.AddResidualBlock(cost_func, nullptr, camera_parameter.data(),
                         points.data());
  }

  ceres::Solver::Options solver_options;
  solver_options.num_threads = 16;
  solver_options.minimizer_progress_to_stdout = true;
  solver_options.max_num_iterations = 500;
  ceres::Solver::Summary summary;

  ceres::Solve(solver_options, &pro, &summary);

  std::cout << summary.FullReport() << std::endl;
  std::cout << "MSE : "
            << std::sqrt(summary.final_cost / problem.observations_.size())
            << std::endl;
}
void CeresRayProblemSolver::Solve(Problem &problem) {
  ceres::Problem pro;
  for (auto &&[pairs, observation] : problem.observations_) {
    CameraParam &camera_parameter = problem.cameras_[pairs.first];
    Landmark &points = problem.points_[pairs.second];
    auto *cost_func = new ceres::AutoDiffCostFunction<RayCostFunction, 3, 9, 3>(
        new RayCostFunction(observation.u(), observation.v()));
    pro.AddResidualBlock(cost_func, nullptr, camera_parameter.data(),
                         points.data());
  }
  ceres::Solver::Options solver_options;
  solver_options.num_threads = 16;
  solver_options.minimizer_progress_to_stdout = true;
  solver_options.max_num_iterations = 500;
  solver_options.linear_solver_type = ceres::LinearSolverType::ITERATIVE_SCHUR;
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &pro, &summary);
  std::cout << summary.BriefReport() << std::endl;
}
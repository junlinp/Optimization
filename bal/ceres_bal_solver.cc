#include "ceres_bal_solver.h"

#include "ceres/autodiff_cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"
#include "ceres/rotation.h"

#include "problem.h"

// camera model see 
// http://grail.cs.washington.edu/projects/bal/
//
struct ProjectFunction {
    ProjectFunction(double u, double v) : u(u), v(v) {}
    double u, v;
    template<class T>
    bool operator()(const T* camera_param,const T* point, T* residual) const {
        T output_point[3];
        ceres::AngleAxisRotatePoint(camera_param, point, output_point);
        output_point[0] += camera_param[3];
        output_point[1] += camera_param[4];
        output_point[2] += camera_param[5];

        output_point[0] /= -output_point[2];
        output_point[1] /= -output_point[2];
        T focal = camera_param[6];
        T K1 = camera_param[7];
        T K2 = camera_param[8];

        T p_norm_2 = output_point[0] * output_point[0] + output_point[1] * output_point[1];
        T distorsion = T(1.0) + K1 * p_norm_2 + K2 * p_norm_2 * p_norm_2;
        residual[0] = T(u) - focal * distorsion * output_point[0];
        residual[1] = T(v) - focal * distorsion * output_point[1];

        return true;
    }
};

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
#include "daba_bal_solver.h"
#include "ceres/problem.h"
#include "ceres/rotation.h"
#include "ceres/solver.h"

#include "cost_function_auto.h"
#include <map>
#include <thread>

#include "Graph/graph_normal_cut.h"
void GraphCut(const Problem &problem,
              std::map<int64_t, int64_t> *cluster_of_camera_index,
              std::map<int64_t, int64_t> *cluster_of_landmark_index) {
  int64_t global_index = 0;
  std::map<int64_t, int64_t> camera_index_to_global_index;
  std::map<int64_t, int64_t> landmark_index_to_global_index;
  for (auto [index_pair, uv] : problem.observations_) {
    int64_t camera_index = index_pair.first;
    if (camera_index_to_global_index.find(camera_index) == camera_index_to_global_index.end()) {
      camera_index_to_global_index[camera_index] = global_index++;
    }
  }

  for (auto [index_pair, uv] : problem.observations_) {
    int64_t camera_index = index_pair.first;
    int64_t landmark_index = index_pair.second;
    if (landmark_index_to_global_index.find(landmark_index) == landmark_index_to_global_index.end()) {
      landmark_index_to_global_index[landmark_index] = global_index++;
    }
  }

  Graph local_graph(global_index);
  for (auto [index_pair, uv] : problem.observations_) {
    int64_t camera_index = index_pair.first;
    int64_t landmark_index = index_pair.second;
    local_graph.SetEdgeValue(camera_index_to_global_index[camera_index],
                             landmark_index_to_global_index[landmark_index],
                             1.0);
  }

  GraphNormalCut cut_solver;
  std::map<int64_t, int64_t> global_index_to_cluster_id;
  auto &&[a, b] = cut_solver.SparseCut(local_graph);

  for (auto index : a) {
    global_index_to_cluster_id[index] = 0;
  }

  for (auto index : b) {
    assert(global_index_to_cluster_id.count(index) == 0);
    global_index_to_cluster_id[index] = 1;
  }

  for (auto [index_pair, uv] : problem.observations_) {
    int64_t camera_index = index_pair.first;
    int64_t landmark_index = index_pair.second;

    int64_t camera_global_index = camera_index_to_global_index[camera_index];
    int64_t landmark_global_index =
        landmark_index_to_global_index[landmark_index];
    (*cluster_of_camera_index)[camera_index] =
        global_index_to_cluster_id[camera_global_index];
    (*cluster_of_landmark_index)[landmark_index] =
        global_index_to_cluster_id[landmark_global_index];
  }
}

void DABAProblemSolver::Solve(Problem &problem) {
  std::map<int64_t, int64_t> cluster_of_camera_index;
  std::map<int64_t, int64_t> cluster_of_landmark_index;
  GraphCut(problem, &cluster_of_camera_index, &cluster_of_landmark_index);

  ceres::Problem global_problem;
  // std::map<int64_t, ceres::Problem> camera_problems;
  // std::map<int64_t, ceres::Problem> landmark_problems;
  std::vector<ceres::Problem> cluster_problems(2);

  for (auto &&[index, camera_parameter] : problem.cameras_) {
    std::array<double, 9> temp_parameter;
    std::copy(camera_parameter.data(), camera_parameter.data() + 9,
              temp_parameter.begin());

    std::copy(temp_parameter.data(), temp_parameter.data() + 9,
              camera_parameters_[index].begin());
    std::copy(temp_parameter.data(), temp_parameter.data() + 9,
              last_camera_parameters_[index].begin());

    int64_t cluster_index = cluster_of_camera_index[index];
    ceres::Problem &problem = cluster_problems[cluster_index];
    problem.AddParameterBlock(camera_parameters_[index].data(), 9);
    problem.AddParameterBlock(last_camera_parameters_[index].data(), 9);
    problem.SetParameterBlockConstant(last_camera_parameters_[index].data());
  }

  for (auto &&[index, landmark] : problem.points_) {
    std::copy(landmark.data(), landmark.data() + 3,
              landmark_position_[index].begin());
    std::copy(landmark.data(), landmark.data() + 3,
              last_landmark_position_[index].begin());

    int64_t cluster_index = cluster_of_landmark_index[index];
    ceres::Problem &problem = cluster_problems[cluster_index];
    problem.AddParameterBlock(landmark_position_[index].data(), 3);
    problem.AddParameterBlock(last_landmark_position_[index].data(), 3);
    problem.SetParameterBlockConstant(last_landmark_position_[index].data());
  }

  for (auto [index_pair, uv] : problem.observations_) {
    int64_t camera_index = index_pair.first;
    int64_t landmark_index = index_pair.second;
    int64_t camera_cluster_id = cluster_of_camera_index[camera_index];
    int64_t landmark_cluster_id = cluster_of_landmark_index[landmark_index];

    if (camera_cluster_id == landmark_cluster_id) {
      ceres::CostFunction *ray_cost_function =
          new ceres::AutoDiffCostFunction<RayCostFunction, 3, 9, 3>(
              new RayCostFunction(uv.u(), uv.v()));
      cluster_problems[camera_cluster_id].AddResidualBlock(
          ray_cost_function, nullptr, camera_parameters_[camera_index].data(),
          landmark_position_[landmark_index].data());
    } else {

      auto &camera_problem = cluster_problems[camera_cluster_id];
      auto &landmark_problem = cluster_problems[landmark_cluster_id];

      ceres::CostFunction *camera_cost_function =
          new ceres::AutoDiffCostFunction<CameraSurrogateCostFunction, 3, 9, 9,
                                          3>(
              new CameraSurrogateCostFunction(uv.u(), uv.v()));
      camera_problem.AddResidualBlock(
          camera_cost_function, nullptr,
          camera_parameters_[camera_index].data(),
          last_camera_parameters_[camera_index].data(),
          last_landmark_position_[landmark_index].data());
      ceres::CostFunction *landmark_cost_function =
          new ceres::AutoDiffCostFunction<LandmarkSurrogatecostFunction, 3, 3,
                                          9, 3>(
              new LandmarkSurrogatecostFunction(uv.u(), uv.v()));
      landmark_problem.AddResidualBlock(
          landmark_cost_function, nullptr,
          landmark_position_[landmark_index].data(),
          last_camera_parameters_[camera_index].data(),
          last_landmark_position_[camera_index].data());
    }
    ceres::CostFunction *ray_cost_function =
        new ceres::AutoDiffCostFunction<RayCostFunction, 3, 9, 3>(
            new RayCostFunction(uv.u(), uv.v()));
    global_problem.AddResidualBlock(ray_cost_function, nullptr,
                                    problem.cameras_[camera_index].data(),
                                    problem.points_[landmark_index].data());
  }
  for (auto& problem : cluster_problems) {
    std::cout << problem.NumResiduals() << std::endl;
  }
  size_t epoch = 0;
  while (epoch++ < 128) {
    std::vector<std::thread> thread_pool;
    for (const auto &[index, parameters] : camera_parameters_) {
      auto &condition_parameters = last_camera_parameters_[index];
      std::copy(parameters.begin(), parameters.end(),
                condition_parameters.begin());
    }
    for (const auto &[index, parameters] : landmark_position_) {
      auto &condition_parameters = last_landmark_position_[index];
      std::copy(parameters.begin(), parameters.end(),
                condition_parameters.begin());
    }
    int indicator = 0;
    for (auto& problem : cluster_problems) {
      auto functor = [&problem, indicator]() {
        ceres::Solver::Options options;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << "Indicator : " << indicator << " Summary: " << summary.BriefReport() << std::endl;
      };
      thread_pool.push_back((std::thread(functor)));
      indicator++;
    }

    for (std::thread &thread : thread_pool) {
      thread.join();
    }
    std::cout << epoch << " epoches" << std::endl;
  }

  for (auto &[index, camera_parameter] : problem.cameras_) {
    std::array<double, 9> temp_parameter;

    std::copy(camera_parameters_[index].begin(),
              camera_parameters_[index].end(), temp_parameter.begin());

    std::copy(temp_parameter.begin(), temp_parameter.end(),
              camera_parameter.data());
  }

  for (auto &[index, landmark] : problem.points_) {
    std::copy(landmark_position_[index].begin(),
              landmark_position_[index].end(), landmark.data());
  }

  // ceres::Solver::Options options;
  // options.max_num_iterations = 512;
  // ceres::Solver::Summary summary;
  // ceres::Solve(options, &global_problem, &summary);
  // std::cout << summary.FullReport() << std::endl;
}
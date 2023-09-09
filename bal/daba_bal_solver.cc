#include "daba_bal_solver.h"

#include <cassert>
#include <ceres/cost_function.h>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <thread>

#include "Graph/graph_normal_cut.h"
#include "bal/daba_subproblem.h"
#include "ceres/problem.h"
#include "ceres/rotation.h"
#include "ceres/solver.h"
#include "cost_function_auto.h"
void GraphCut(const Problem &problem,
              std::map<int64_t, int64_t> *cluster_of_camera_index,
              std::map<int64_t, int64_t> *cluster_of_landmark_index) {
  int64_t global_index = 0;
  std::map<int64_t, int64_t> camera_index_to_global_index;
  std::map<int64_t, int64_t> landmark_index_to_global_index;
  for (auto [index_pair, uv] : problem.observations_) {
    int64_t camera_index = index_pair.first;
    if (camera_index_to_global_index.find(camera_index) ==
        camera_index_to_global_index.end()) {
      camera_index_to_global_index[camera_index] = global_index++;
    }
  }

  for (auto [index_pair, uv] : problem.observations_) {
    // int64_t camera_index = index_pair.first;
    int64_t landmark_index = index_pair.second;
    if (landmark_index_to_global_index.find(landmark_index) ==
        landmark_index_to_global_index.end()) {
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

void RandomGraphCut(const Problem &problem, int parition,
                    std::map<int64_t, int64_t> *cluster_of_camera_index,
                    std::map<int64_t, int64_t> *cluster_of_landmark_index) {
  std::uniform_int_distribution<int64_t> distribution(0, parition - 1);
  std::mt19937 generator(13);

  assert(cluster_of_camera_index != nullptr);
  assert(cluster_of_landmark_index != nullptr);

  for (auto [index_pair, uv] : problem.observations_) {
    int64_t camera_index = index_pair.first;
    if (cluster_of_camera_index->count(camera_index) == 0) {
      (*cluster_of_camera_index)[camera_index] = distribution(generator);
    }
    int64_t landmark_index = index_pair.second;
    if (cluster_of_landmark_index->count(landmark_index) == 0) {
      (*cluster_of_landmark_index)[landmark_index] = distribution(generator);
    }
  }
}

template <class T>
void NesterovStep(const T &last_parameters, const T &current_parameters,
                  double grama, T *next_step) {
  assert(last_parameters.size() == current_parameters.size());
  size_t n = last_parameters.size();
  for (size_t i = 0; i < n; i++) {
    (*next_step)[i] = current_parameters[i] +
                      grama * (current_parameters[i] - last_parameters[i]);
  }
}

double s(int k) {
  static std::map<int, double> cache;
  if (k == 0) {
    cache[0] = 1.0;
  }
  if (cache.find(k) == cache.end()) {
    cache[k] = 0.5 * (std::sqrt(4.0 * s(k - 1) * s(k - 1) + 1) + 1);
  }
  return cache[k];
}

void DABAProblemSolver::Solve(Problem &problem) {
  std::map<int64_t, int64_t> cluster_of_camera_index;
  std::map<int64_t, int64_t> cluster_of_landmark_index;
  const int partition = 2;
  RandomGraphCut(problem, partition, &cluster_of_camera_index,
                 &cluster_of_landmark_index);

  std::vector<std::shared_ptr<DabaSubproblem>> cluster_subproblems;
  for (int i = 0; i < partition; i++) {
    cluster_subproblems.push_back(std::make_shared<DabaSubproblem>(i));
  }


  std::map<int64_t, int64_t> camera_boardcast_map;
  std::map<int64_t, int64_t> point_boardcast_map;

  for (auto [index_pair, uv] : problem.observations_) {
    int64_t camera_index = index_pair.first;
    int64_t landmark_index = index_pair.second;
    int64_t camera_cluster_id = cluster_of_camera_index[camera_index];
    int64_t landmark_cluster_id = cluster_of_landmark_index[landmark_index];

    if (camera_cluster_id == landmark_cluster_id) {
      cluster_subproblems[camera_cluster_id]->AddInternalEdge(
          camera_index, problem.cameras_.at(camera_index).array(),
          landmark_index, problem.points_.at(landmark_index).array(),
          {uv.u(), uv.v()});

    } else {
      cluster_subproblems[camera_cluster_id]->AddCamera(
          camera_index, problem.cameras_.at(camera_index).array(),
          landmark_index, problem.points_.at(landmark_index).array(),
          {uv.u(), uv.v()});

      cluster_subproblems[landmark_cluster_id]->AddPoint(
          landmark_index, problem.points_.at(landmark_index).array(),
          camera_index, problem.cameras_.at(camera_index).array(),
          {uv.u(), uv.v()});

      camera_boardcast_map[camera_index] = landmark_cluster_id;
      point_boardcast_map[landmark_index] = camera_cluster_id;
    }
  }

  auto boardcast_functor =
      [&camera_boardcast_map, &point_boardcast_map, &cluster_subproblems](
          int iteration,
          std::map<int64_t, std::array<double, 9>> camera_parameters,
          std::map<int64_t, std::array<double, 3>> point_parameters) {
        using CameraMap = std::map<int64_t, std::array<double, 9>>;
        using PointMap = std::map<int64_t, std::array<double, 3>>;
        std::map<int64_t, CameraMap> cluster_camera_map;
        std::map<int64_t, PointMap> cluster_point_map;
        
        for (auto&& pair : camera_parameters) {
          int64_t cluster_id = camera_boardcast_map.at(pair.first);
          cluster_camera_map[cluster_id].insert(pair);
        }

        for (auto&& pair : point_parameters) {
          int64_t cluster_id = point_boardcast_map.at(pair.first);
          cluster_point_map[cluster_id].insert(pair);
        }

        for (auto& cluster_problem : cluster_subproblems) {
          cluster_problem->ReceiveExternalParameters(
              iteration,
              cluster_camera_map[cluster_problem->ClusterId()],
              cluster_point_map[cluster_problem->ClusterId()]);
        }
      };

  for (auto &p : cluster_subproblems) {
    p->SetBoardcastCallback(boardcast_functor);
  }

  for (auto &p : cluster_subproblems) {
    p->Start();
    std::cout << "cluster " << p->ClusterId() << " start" << std::endl;
  }
  std::set<int64_t> camera_visited;
  std::set<int64_t> point_visited;
  for (auto &cluster_subproblem : cluster_subproblems) {
    cluster_subproblem->WaitForFinish();
    for (auto pair : cluster_subproblem->ClusterCameraData()) {
      std::copy(pair.second.begin(), pair.second.end(),
                problem.cameras_.at(pair.first).data());
      if (camera_visited.count(pair.first) == 0) {
        camera_visited.insert(pair.first);
      } else {
        std::cout << "Error camera" << std::endl;
      }
    }
    for (auto pair : cluster_subproblem->ClusterPointData()) {
      std::copy(pair.second.begin(), pair.second.end(),
                problem.points_.at(pair.first).data());
      if (point_visited.count(pair.first) == 0) {
        point_visited.insert(pair.first);
      } else {
        std::cout << "Error camera" << std::endl;
      }
    }
  }
  
  // assert(camera_visited.size() == problem.cameras_.size());
  // assert(point_visited.size() == problem.points_.size());

  //   if (camera_visited.find(camera_index) == camera_visited.end()) {
  //     camera_visited.insert(camera_index);

  //     ceres::Problem& problem = cluster_problems[camera_cluster_id];
  //     auto *costfunction_ptr =
  //         new ceres::AutoDiffCostFunction<WeightVectorDiff<9>, 9, 9>(
  //             new WeightVectorDiff<9>(
  //                 last_camera_parameters_[camera_index].data(), 0.1));
  //     problem.AddResidualBlock(costfunction_ptr, nullptr,
  //                              camera_parameters_[camera_index].data());
  //   }

  //   if (landmark_visited.find(landmark_index) == landmark_visited.end()) {
  //     landmark_visited.insert(landmark_index);
  //     ceres::Problem& problem = cluster_problems[landmark_cluster_id];
  //     auto *costfunction_ptr =
  //         new ceres::AutoDiffCostFunction<WeightVectorDiff<3>, 3, 3>(
  //             new WeightVectorDiff<3>(
  //                 last_landmark_position_[landmark_index].data(), 0.1));
  //                 problem.AddResidualBlock(costfunction_ptr, nullptr,
  //                 landmark_position_[landmark_index].data());
  //   }
  // }

  // for (auto& problem : cluster_problems) {
  //   std::cout << problem.NumResiduals() << std::endl;
  // }
  // size_t epoch = 0;
  // std::map<int64_t, std::array<double, 9>> last_camera_parameters =
  // camera_parameters_; std::map<int64_t, std::array<double, 3>>
  // last_landmark_position = landmark_position_; int grama_index = 0; double
  // last_error = std::numeric_limits<double>::max(); while (epoch++ < 1024) {
  //   std::vector<std::thread> thread_pool;
  //   double grama = (s(grama_index) -1) / s(grama_index+ 1);
  //   for (const auto &[index, parameters] : camera_parameters_) {
  //     auto &condition_parameters = last_camera_parameters_[index];

  //     NesterovStep(last_camera_parameters[index], parameters, grama,
  //     &condition_parameters); std::copy(parameters.begin(), parameters.end(),
  //               last_camera_parameters[index].begin());
  //   }

  //   for (const auto &[index, parameters] : landmark_position_) {
  //     auto &condition_parameters = last_landmark_position_[index];
  //     NesterovStep(last_landmark_position[index], parameters, grama,
  //     &condition_parameters); std::copy(parameters.begin(), parameters.end(),
  //               last_landmark_position[index].begin());
  //   }

  //   int indicator = 0;
  //   double current_error = 0.0;
  //   std::mutex current_error_mutex;
  //   for (auto& problem : cluster_problems) {
  //     auto functor = [&problem, indicator, &current_error_mutex,
  //     &current_error]() {
  //       ceres::Solver::Options options;
  //       options.max_num_iterations = 500;
  //       ceres::Solver::Summary summary;
  //       ceres::Solve(options, &problem, &summary);
  //       std::cout << "Indicator : " << indicator << " Summary: " <<
  //       summary.BriefReport() << std::endl;
  //       {
  //         std::lock_guard<std::mutex> lk_(current_error_mutex);
  //         current_error += summary.final_cost;
  //       }
  //     };
  //     thread_pool.push_back((std::thread(functor)));
  //     indicator++;
  //   }

  //   for (std::thread &thread : thread_pool) {
  //     thread.join();
  //   }

  //   if (current_error >= last_error) {
  //     grama_index++;
  //     for (const auto &[index, parameters] : camera_parameters_) {
  //       auto &condition_parameters = last_camera_parameters_[index];
  //       auto &pre_condition_parameters = last_camera_parameters[index];
  //       std::copy(pre_condition_parameters.begin(),
  //       pre_condition_parameters.end(), condition_parameters.begin());
  //     }

  //     for (const auto &[index, parameters] : landmark_position_) {
  //       auto &condition_parameters = last_landmark_position_[index];
  //       auto& pre_condition_parameters = last_landmark_position[index];
  //       std::copy(pre_condition_parameters.begin(),
  //                 pre_condition_parameters.end(),
  //                 condition_parameters.begin());
  //     }
  //     thread_pool.clear();
  //     current_error = 0;
  //     for (auto &problem : cluster_problems) {
  //       auto functor = [&problem, &current_error_mutex, &current_error]() {
  //         ceres::Solver::Options options;
  //         options.max_num_iterations = 500;
  //         ceres::Solver::Summary summary;
  //         ceres::Solve(options, &problem, &summary);

  //         {
  //           std::lock_guard<std::mutex> lk_(current_error_mutex);
  //           current_error += summary.final_cost;
  //         }
  //       };
  //       thread_pool.push_back((std::thread(functor)));
  //     }
  //     for (std::thread &thread : thread_pool) {
  //       thread.join();
  //     }
  //   } else {
  //     grama_index++;
  //   }
  //   std::cout << epoch << " epoches." << last_error << " -> " <<
  //   current_error << std::endl; last_error = current_error;
  // }

  // for (auto &[index, camera_parameter] : problem.cameras_) {
  //   std::array<double, 9> temp_parameter;

  //   std::copy(camera_parameters_[index].begin(),
  //             camera_parameters_[index].end(), temp_parameter.begin());

  //   std::copy(temp_parameter.begin(), temp_parameter.end(),
  //             camera_parameter.data());
  // }

  // for (auto &[index, landmark] : problem.points_) {
  //   std::copy(landmark_position_[index].begin(),
  //             landmark_position_[index].end(), landmark.data());
  // }

  
  // ceres::Solver::Options options;
  // options.max_num_iterations = 2;
  // ceres::Solver::Summary summary;
  // ceres::Solve(options, &global_problem, &summary);
  // std::cout << "global : " << summary.FullReport() << std::endl;
}
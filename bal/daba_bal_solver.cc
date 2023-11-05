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
  const int partition = 2.0 * std::thread::hardware_concurrency();
  RandomGraphCut(problem, partition, &cluster_of_camera_index,
                 &cluster_of_landmark_index);

  std::vector<std::shared_ptr<DabaSubproblem>> cluster_subproblems;
  for (int i = 0; i < partition; i++) {
    cluster_subproblems.push_back(std::make_shared<DabaSubproblem>(i));
  }


  std::map<int64_t, std::vector<int64_t>> camera_boardcast_map;
  std::map<int64_t, std::vector<int64_t>> point_boardcast_map;

 
  std::vector<std::tuple<int64_t, int64_t, std::array<double, 2>>> surrogate_edges;
  std::set<int64_t> surrogate_camera_index;
  std::set<int64_t> surrogate_point_index;

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

      camera_boardcast_map[camera_index].push_back( landmark_cluster_id);
      point_boardcast_map[landmark_index].push_back(camera_cluster_id);

      surrogate_edges.push_back(std::make_tuple(
          camera_index, landmark_index, std::array<double, 2>{uv.u(), uv.v()}));
      surrogate_camera_index.insert(camera_index);
      surrogate_point_index.insert(landmark_index);
    }
  }

  using CameraMap = std::map<int64_t, std::array<double, 9>>;
  using PointMap = std::map<int64_t, std::array<double, 3>>;

  std::map<int64_t, CameraMap> iteration_camera_parameters;
  std::map<int64_t, PointMap> iteration_point_parameters;
  std::mutex iteration_parameters_mutex;

  auto boardcast_functor =
      [&camera_boardcast_map, &point_boardcast_map, &cluster_subproblems,
       &surrogate_camera_index, &surrogate_point_index, &surrogate_edges,
       &iteration_camera_parameters, &iteration_point_parameters, &iteration_parameters_mutex](
          int iteration,
          std::map<int64_t, std::array<double, 9>> camera_parameters,
          std::map<int64_t, std::array<double, 3>> point_parameters) {

        using CameraMap = std::map<int64_t, std::array<double, 9>>;
        using PointMap = std::map<int64_t, std::array<double, 3>>;
        std::map<int64_t, CameraMap> cluster_camera_map;
        std::map<int64_t, PointMap> cluster_point_map;

        {
          // std::lock_guard<std::mutex> lk_lock(iteration_parameters_mutex);
          for (auto pair : camera_parameters) {
            // iteration_camera_parameters[iteration].insert(pair);
            for (int64_t cluster_id : camera_boardcast_map.at(pair.first)) {
              cluster_camera_map[cluster_id].insert(pair);
            }
          }

          for (auto pair : point_parameters) {
            // iteration_point_parameters[iteration].insert(pair);
            for (int64_t cluster_id : point_boardcast_map.at(pair.first)) {
              cluster_point_map[cluster_id].insert(pair);
            }
          }
        }

        for (auto &cluster_problem : cluster_subproblems) {
          std::thread(
              [iteration, cluster_problem,
               c_p = cluster_camera_map[cluster_problem->ClusterId()],
               p_p = cluster_point_map[cluster_problem->ClusterId()]]() {
                cluster_problem->ReceiveExternalParameters(iteration, c_p, p_p);
              }).detach();
        }

        // if (iteration_camera_parameters[iteration].size() ==
        //         surrogate_camera_index.size() &&
        //     iteration_point_parameters[iteration].size() ==
        //         surrogate_point_index.size()) {
            
        //     double cost_value = 0.0;
        //     for (auto edge : surrogate_edges) {
        //       int64_t camera_index = std::get<0>(edge);
        //       int64_t point_index = std::get<1>(edge);
        //       std::array<double, 2> uv = std::get<2>(edge);
        //       RayCostFunction function(uv[0], uv[1]);
        //       double residual[3];
        //       function(
        //           iteration_camera_parameters[iteration][camera_index].data(),
        //           iteration_point_parameters[iteration][point_index].data(),
        //           residual);
        //       cost_value += 0.5 * (residual[0] * residual[1] + residual[1] * residual[1] + residual[2] * residual[2]);
        //     }

        //     std::cout << "[" << iteration << "] Total surrogate : " << cost_value << " Mean : " << cost_value / surrogate_edges.size() << std::endl;
        // }
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
}
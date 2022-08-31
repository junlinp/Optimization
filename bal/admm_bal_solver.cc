#include "admm_bal_solver.h"
#include "ceres_bal_solver.h"

#include <map>
#include <set>

#include "Graph/graph.h"
#include "Graph/graph_normal_cut.h"

void ADMMProblemSolver::Solve(Problem &problem) {

  std::set<size_t> camera_ids;

  for (auto iter : problem.cameras_) {
    camera_ids.insert(iter.first);
  }

  std::map<size_t, std::vector<size_t>> camera_see_points;

  for (auto &&[piar, _] : problem.observations_) {
    camera_see_points[piar.first].push_back(piar.second);
  }

  Graph graph(problem.points_.size());
  std::cout << "Graph Setting" << std::endl;
  for (auto &&[_, point_in_common_camera] : camera_see_points) {
    for (size_t i = 0; i < point_in_common_camera.size(); i++) {
      for (size_t j = i + 1; j < point_in_common_camera.size(); j++) {
        double edge_value = graph.GetEdgeValue(point_in_common_camera[i],
                                               point_in_common_camera[j]);
        graph.SetEdgeValue(point_in_common_camera[i], point_in_common_camera[j],
                           edge_value + 1.0);
      }
    }
  }

  std::cout << "Graph Generation finish" << std::endl;

  GraphNormalCut cut_solver;
  auto&& [a_set, b_set] = cut_solver.Cut(graph);
  std::cout << "Graph Cut finish " << std::endl;

  auto local_camera = problem.cameras_;
  auto& points = problem.points_;
  auto local_observation = problem.observations_;

  for(auto&& [camera_id, see_points] : camera_see_points) {
    size_t augmented_camera_id = local_camera.size();
    for(size_t point_id : see_points) {
        if (b_set.find(point_id) != b_set.end()) {
            local_camera[augmented_camera_id] = local_camera[camera_id];

            local_observation[{augmented_camera_id, point_id}] = local_observation[{camera_id, point_id}];
            local_observation.erase({camera_id, point_id});
        }
    }
  }
  std::cout << "A set : " << a_set.size() << ", B set : " << b_set.size() << std::endl;

}
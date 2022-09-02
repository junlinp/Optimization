#include "admm_bal_solver.h"
#include "bal/problem.h"
#include "ceres_bal_solver.h"

#include <map>
#include <set>
#include <unordered_map>

#include "ceres/problem.h"
#include "ceres/rotation.h"

#include "Graph/graph.h"
#include "Graph/graph_normal_cut.h"

Eigen::VectorXd Projector(Eigen::VectorXd);

class BlockSolver {
  BlockSolver(
      std::unordered_map<size_t, CameraParam> &camera_parameters,
      std::unordered_map<size_t, Landmark> &points_parameters,
      std::unordered_map<std::pair<size_t, size_t>, Observation> &observations,
      std::set<size_t> point_idx_belongs_blocks)
      :camera_se3_intrinsic_parameters_(), points_parameters_(points_parameters), observations_(observations), problem_() {

        for(auto&& [pairs, observation] : observations_) {

            size_t camera_id = pairs.first;
            size_t point_id = pairs.second;

            if (camera_se3_intrinsic_parameters_.find(camera_id) == camera_se3_intrinsic_parameters_.end()) {
                double* data_ptr = camera_parameters.at(camera_id).data();

                // 9 + 3 + 3
                double vector_ptr[15];
                std::memcpy(vector_ptr + 6, data_ptr, sizeof(double) * 9);
                ceres::AngleAxisToRotationMatrix(data_ptr, vector_ptr);
                Eigen::VectorXd se3_intrinsic_parameter = Eigen::Map<Eigen::VectorXd>(vector_ptr, 15);
                camera_se3_intrinsic_parameters_[camera_id] = se3_intrinsic_parameter;
            }
            // create cost function
            ceres::CostFunction* cost_function = nullptr;
            problem_.AddResidualBlock(cost_function, nullptr, camera_se3_intrinsic_parameters_[camera_id].data(), points_parameters_[point_id].data());
        }
      }



  std::unordered_map<size_t, Eigen::VectorXd> camera_se3_intrinsic_parameters_;
  std::unordered_map<size_t, Landmark> &points_parameters_;
  std::unordered_map<std::pair<size_t, size_t>, Observation> &observations_;
  ceres::Problem problem_;

};

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
  auto &&[a_set, b_set] = cut_solver.Cut(graph);
  std::cout << "Graph Cut finish " << std::endl;

  auto local_camera = problem.cameras_;
  auto &points = problem.points_;
  auto local_observation = problem.observations_;

  std::map<size_t, size_t> augmented_camera_id_to_consensus;

  for (auto &&[camera_id, see_points] : camera_see_points) {
    size_t augmented_camera_id = local_camera.size();
    augmented_camera_id_to_consensus[camera_id] = camera_id;
    for (size_t point_id : see_points) {
      if (b_set.find(point_id) != b_set.end()) {
        local_camera[augmented_camera_id] = local_camera[camera_id];

        local_observation[{augmented_camera_id, point_id}] =
            local_observation[{camera_id, point_id}];
        local_observation.erase({camera_id, point_id});
        augmented_camera_id_to_consensus[augmented_camera_id] = camera_id;
      }
    }
  }

  std::cout << "A set : " << a_set.size() << ", B set : " << b_set.size()
            << std::endl;

  // TODO(junlinp@deepmirror.com):
  //
  // two-block
  //
  //
  // (1) create a Block Solver
  // (2) for each iterate
  //     a Block unconstrainted solve
  //     b shared the camera parameter
  //     c project the camera parameter to consensus
  //     d distribute the consensus to each block solver
  // (3) camera consensus and point solution as final result
  //
  //  BLock Solver (camera_parameters, point_parameters, observations,
  //  point_ids_belongs_to_this_block)
  //
  //  std::map<camera_id, CameraParameterBlocks> = GetCmeraBLocks
  //  SetCameraConsensus
  //  SetDualVariable
  //  GetPoints?(optmize it on momery directly)?
  //
  //
  //  stop criteria
  //
}
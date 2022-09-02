#include "admm_bal_solver.h"
#include "bal/problem.h"
#include "ceres_bal_solver.h"

#include <Eigen/src/Core/Matrix.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/cost_function.h>
#include <map>
#include <set>

#include "ceres/problem.h"
#include "ceres/rotation.h"

#include "Graph/graph.h"
#include "Graph/graph_normal_cut.h"

Eigen::VectorXd Projector(Eigen::VectorXd);

struct BlockFunctor {
  BlockFunctor(double u, double v) : u_(u), v_(v) {}
  double u_, v_;

  template <class T>
  bool operator()(const T *camera_parameters, const T *point_parameters,
                  T *residual) const {
    Eigen::Matrix<T, 3, 3> rotation =
        Eigen::Map<Eigen::Matrix<T, 3, 3>>(camera_parameters, 9);
    using VectorMap = Eigen::Map<Eigen::Vector<T, 3>>;

    Eigen::Vector<T, 3> point = rotation * VectorMap(point_parameters, 3) +
                                VectorMap(camera_parameters + 9, 3);
    T x = point(0) / point(2);
    T y = point(1) / point(2);

    T focal = camera_parameters[12];
    T K1 = camera_parameters[13];
    T K2 = camera_parameters[14];

    T p_norm_2 = x * x + y * y;
    T distorsion = T(1.0) + K1 * p_norm_2 + K2 * p_norm_2 * p_norm_2;

    residual[0] = T(u_) - focal * distorsion * x;
    residual[1] = T(v_) - focal * distorsion * y;
    return true;
  }
};

struct DualTermFunctor {
  DualTermFunctor(Eigen::VectorXd &w, Eigen::VectorXd &y) : w_(w), y_(y) {}

  template <class T> bool operator()(const T *x, T *residual) const {
    residual[0] = T(0.0);
    for (int i = 0; i < 15; i++) {
      residual[0] += T(w_(i)) * (x[i] - T(y_(i)));
    }
    return true;
  }

  Eigen::VectorXd &w_;
  Eigen::VectorXd &y_;
};

struct RegularTermFunctor {
  RegularTermFunctor(Eigen::VectorXd &y, double beta) : y_(y), beta_(beta) {}

  template <class T> bool operator()(const T *x, T *residual) const {
    using VectorMap = Eigen::Map<Eigen::Vector<T, 15>>;
    double coefficent = std::sqrt(beta_ / 2.0);

    T y[15];
    for (int i = 0; i < 15; i++) {
      y[i] = T(y_(i));
    }

    VectorMap(residual, 15) =
        coefficent * (VectorMap(x, 15) - VectorMap(y, 15));
    return true;
  }

  Eigen::VectorXd &y_;
  double beta_;
};

class BlockSolver {
  public:
  BlockSolver(std::map<size_t, CameraParam> &camera_parameters,
              std::map<size_t, Landmark> &points_parameters,
              std::map<std::pair<size_t, size_t>, Observation> &observations,
              std::set<size_t> point_idx_belongs_blocks,
              std::map<size_t, Eigen::VectorXd> local_camera_consensus_variable,
              std::map<size_t, Eigen::VectorXd> local_dual_variable,
              std::map<size_t, size_t> camera_id_map_consensus_id,
              double beta = 1.0)
      : camera_se3_intrinsic_parameters_(),
        local_camera_consensus_variable_(local_camera_consensus_variable),
        local_dual_variable_(local_dual_variable),
        camera_id_map_consensus_id_(camera_id_map_consensus_id), beta_(beta),
        points_parameters_(points_parameters), observations_(observations),
        problem_() {

    for (auto &&[pairs, observation] : observations_) {

      size_t camera_id = pairs.first;
      size_t point_id = pairs.second;

      if (point_idx_belongs_blocks.find(point_id) !=
          point_idx_belongs_blocks.end()) {

        if (camera_se3_intrinsic_parameters_.find(camera_id) ==
            camera_se3_intrinsic_parameters_.end()) {
          double *data_ptr = camera_parameters.at(camera_id).data();

          // 9 + 3 + 3
          double vector_ptr[15];
          std::memcpy(vector_ptr + 6, data_ptr, sizeof(double) * 9);
          ceres::AngleAxisToRotationMatrix(data_ptr, vector_ptr);
          Eigen::VectorXd se3_intrinsic_parameter =
              Eigen::Map<Eigen::VectorXd>(vector_ptr, 15);
          camera_se3_intrinsic_parameters_[camera_id] = se3_intrinsic_parameter;
        }
        // cost function
        ceres::CostFunction *cost_function =
            new ceres::AutoDiffCostFunction<BlockFunctor, 2, 15, 3>(
                new BlockFunctor(observation.u(), observation.v()));
        problem_.AddResidualBlock(
            cost_function, nullptr,
            camera_se3_intrinsic_parameters_[camera_id].data(),
            points_parameters_[point_id].data());

        // dual term
        // |<W_i, X_i - Y>|^2
        ceres::CostFunction *dual_term =
            new ceres::AutoDiffCostFunction<DualTermFunctor, 1, 15>(
                new DualTermFunctor(
                    local_dual_variable_[camera_id],
                    local_camera_consensus_variable_
                        [camera_id_map_consensus_id_[camera_id]]));
        problem_.AddResidualBlock(
            dual_term, nullptr,
            camera_se3_intrinsic_parameters_[camera_id].data());
        // regular term
        // beta / 2.0 * | X_i - Y | ^ 2

        ceres::CostFunction *regular_term =
            new ceres::AutoDiffCostFunction<RegularTermFunctor, 15, 15>(
                new RegularTermFunctor(
                    local_camera_consensus_variable_
                        [camera_id_map_consensus_id_[camera_id]],
                    beta_));

        problem_.AddResidualBlock(
            regular_term, nullptr,
            camera_se3_intrinsic_parameters_[camera_id].data());
      }
    }
  }

  void Optmization();

  std::map<size_t, Eigen::VectorXd> GetLocalCameraParameters() const {
    return camera_se3_intrinsic_parameters_;
  }

  void UpdateConsensusAndDualVarialbe(
      const std::map<size_t, Eigen::VectorXd> &consensus_varialbe,
      const std::map<size_t, Eigen::VectorXd> &dual_variable) {
    local_camera_consensus_variable_ = consensus_varialbe;
    local_dual_variable_ = dual_variable;
  }
private:
  std::map<size_t, Eigen::VectorXd> camera_se3_intrinsic_parameters_;
  std::map<size_t, Eigen::VectorXd> local_camera_consensus_variable_;
  std::map<size_t, Eigen::VectorXd> local_dual_variable_;
  std::map<size_t, size_t> camera_id_map_consensus_id_;
  double beta_;

  std::map<size_t, Landmark> &points_parameters_;
  std::map<std::pair<size_t, size_t>, Observation> &observations_;
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
  std::map<size_t, Eigen::VectorXd> consensus_variable, dual_variable;
  for (auto &&[camera_id, _] : local_camera) {
    consensus_variable[camera_id] = Eigen::VectorXd::Random(15);
  }

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

  for (auto &&[camera_id, _] : local_camera) {
    dual_variable[camera_id] = Eigen::VectorXd::Random(15);
  }

  std::cout << "A set : " << a_set.size() << ", B set : " << b_set.size()
            << std::endl;

  //
  //
  // TODO(junlinp@deepmirror.com):
  //
  // two-block
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

  double beta = 1.0;

  BlockSolver ASolver(local_camera, points, local_observation, a_set,
                      consensus_variable, dual_variable,
                      augmented_camera_id_to_consensus, beta);
  BlockSolver BSolver(local_camera, points, local_observation, b_set,
                      consensus_variable, dual_variable,
                      augmented_camera_id_to_consensus, beta);


  for(int epoch = 0; epoch < 1024; epoch++) {
    // actually, there can parallel due to there are no common parameter bewteen ASolver and BSolver.
    ASolver.Optmization();
    BSolver.Optmization();

    // Gather data

    auto a_local_camera = ASolver.GetLocalCameraParameters();
    auto b_local_camera = ASolver.GetLocalCameraParameters();

    auto temp_camera = a_local_camera;
    for (auto&& [camera_id, data] : temp_camera) {
      if (b_set.find(camera_id) != b_set.end()) {
        data = b_local_camera[camera_id];
      }
    }

    // TODO
    //
    // Get and Project to manifold
    // consensus_variable;
    //

    for(auto&& [consensus_camera_id, consensus_camera_value] : consensus_variable) {
      size_t num = 0;
      Eigen::VectorXd camera_sum = Eigen::VectorXd::Zero(15);
      Eigen::VectorXd dual_sum = Eigen::VectorXd::Zero(15);
      for (auto&& [augment_camera_id, consensus_camera_id_] : augmented_camera_id_to_consensus) {
        if (consensus_camera_id_ == consensus_camera_id) {
          num++;
          camera_sum += temp_camera[augment_camera_id];
          dual_sum += dual_variable[augment_camera_id];
        }
      }
      consensus_variable[consensus_camera_id] = ProjectOperator(camera_sum / num + beta / num * dual_sum);
    }


    // Update dual varialbe
    //
    //
    ASolver.UpdateConsensusAndDualVarialbe(consensus_variable, dual_variable);
    BSolver.UpdateConsensusAndDualVarialbe(consensus_variable, dual_variable);
  }
}
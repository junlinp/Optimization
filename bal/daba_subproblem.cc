#include "daba_subproblem.h"

#include <ceres/autodiff_cost_function.h>

#include <chrono>
#include <fstream>
#include <mutex>
#include <string>

#include "ceres/solver.h"
#include "cost_function_auto.h"
namespace {
template <class T, int DIM>
void NesteorvStep(const T *previous, T *current, double nesteorv_coeeficient) {
  for (int i = 0; i < DIM; i++) {
    current[i] = current[i] + nesteorv_coeeficient * (current[i] - previous[i]);
  }
}

}  // namespace

DabaSubproblem::DabaSubproblem(int cluster_id) : cluster_id_(cluster_id) {}

void DabaSubproblem::AddInternalEdge(
    int64_t camera_id, const std::array<double, 9> &camera_parameters,
    int64_t point_id, const std::array<double, 3> &point_parameters,
    std::array<double, 2> uv) {
  camera_parameters_.insert_or_assign(camera_id, camera_parameters);
  point_parameters_[point_id] = point_parameters;

  ceres::CostFunction *ray_cost_function =
      new ceres::AutoDiffCostFunction<RayCostFunction, 3, 9, 3>(
          new RayCostFunction(uv[0], uv[1]));

  problem_.AddResidualBlock(ray_cost_function, nullptr,
                            camera_parameters_[camera_id].data(),
                            point_parameters_[point_id].data());
}

void DabaSubproblem::AddCamera(
    int64_t camera_id, const std::array<double, 9> &camera_parameters,
    int64_t external_point_id,
    const std::array<double, 3> &external_point_parameters,
    const std::array<double, 2> &uv) {
  camera_parameters_[camera_id] = camera_parameters;
  previous_external_camera_[camera_id] = camera_parameters;
  condition_external_camera_[camera_id] = camera_parameters;
  // it should be update by ReceiveExternalPoint
  external_other_point_[external_point_id] = external_point_parameters;

  ceres::CostFunction *camera_cost_function =
      new ceres::AutoDiffCostFunction<CameraSurrogateCostFunction, 3, 9>(
          new CameraSurrogateCostFunction(
              condition_external_camera_[camera_id].data(),
              external_other_point_[external_point_id].data(), uv[0], uv[1]));

  problem_.AddResidualBlock(camera_cost_function, nullptr,
                            camera_parameters_[camera_id].data());
}

void DabaSubproblem::AddPoint(
    int64_t point_id, const std::array<double, 3> &point_parameters,
    int64_t external_camera_id,
    const std::array<double, 9> &external_camera_parameters,
    const std::array<double, 2> &uv) {
  point_parameters_.insert_or_assign(point_id, point_parameters);
  previous_external_point_.insert_or_assign(point_id, point_parameters);
  condition_external_point_.insert_or_assign(point_id, point_parameters);

  external_other_camera_[external_camera_id] = external_camera_parameters;

  ceres::CostFunction *landmark_cost_function =
      new ceres::AutoDiffCostFunction<LandmarkSurrogatecostFunction, 3, 3>(
          new LandmarkSurrogatecostFunction(
              external_other_camera_.at(external_camera_id).data(),
              condition_external_point_.at(point_id).data(), uv[0], uv[1]));
  problem_.AddResidualBlock(landmark_cost_function, nullptr,
                            point_parameters_.at(point_id).data());
}

void DabaSubproblem::ReceiveExternalParameters(
    int iteration,
    const std::map<int64_t, CameraParameters> &camera_external_parameters,
    const std::map<int64_t, PointParameters> &point_external_parameters) {

  std::lock_guard<std::mutex> lk_k(external_queue_mutex_);
  for(auto camera_pair : camera_external_parameters) {
    external_other_camera_queue_.push(std::make_tuple(iteration, camera_pair.first, camera_pair.second));
  }
  for(auto point_pair :point_external_parameters) {
    external_other_point_queue_.push(
        std::make_tuple(iteration, point_pair.first, point_pair.second));
  }
}

void DabaSubproblem::Start() {
  for (auto [camera_index, camera_parameters] : camera_parameters_) {
    if (previous_external_camera_.count(camera_index) == 0) {
      previous_internal_camera_.insert({camera_index, camera_parameters});
    }

    // ceres::CostFunction *norm_function =
    //     new ceres::AutoDiffCostFunction<WeightVectorDiff<9>, 9, 9>(
    //         new WeightVectorDiff<9>(
    //             previous_internal_camera_[camera_index].data(), 0.5));

    // problem_.AddResidualBlock(norm_function, nullptr,
    // camera_parameters.data());
  }

  for (auto [point_index, point_parameters] : point_parameters_) {
    if (previous_external_point_.count(point_index) == 0) {
      previous_internal_point_.insert({point_index, point_parameters});
    }

    // ceres::CostFunction *norm_function =
    //     new ceres::AutoDiffCostFunction<WeightVectorDiff<3>, 3, 3>(
    //         new WeightVectorDiff<3>(
    //             previous_internal_point_[point_index].data(), 0.5));

    // problem_.AddResidualBlock(norm_function, nullptr,
    // point_parameters.data());
  }
  auto solve_functor = [this]() {
    constexpr int max_iteration = 128;
    iteration_ = 0;

    double s = 1;
    double last_cost_value = std::numeric_limits<double>::max();
    std::ofstream ofs(std::to_string(cluster_id_) + ".log");
    while (iteration_++ < max_iteration) {
      ofs << iteration_ << " CollectionNeighborParameters()" << std::endl;
      CollectionNeighborParameters();
      ofs << iteration_ << " CollectionNeighborParameters() finish" << std::endl;
      double s_next = (std::sqrt(4 * s * s + 1) + 1) * 0.5;
      double nesteorv_coeeficient = (s - 1) / s_next;
      s = s_next;

      ceres::Solver::Summary summary;
      ceres::Solver::Options options;
      options.max_num_iterations = 512;
      ceres::Solve(options, &problem_, &summary);
      // std::cout << "cluster " << cluster_id_ << " iteration " << iteration_
                // << " summary : " << summary.BriefReport() << std::endl;
      ofs << "cluster " << cluster_id_ << " iteration " << iteration_
                << " summary : " << summary.BriefReport() << std::endl;

      if (summary.final_cost > last_cost_value) {
        std::cout << "cluster " << cluster_id_ << " restart." << iteration_
                  << "cost increase from " << last_cost_value << " to "
                  << summary.final_cost << std::endl;
        // s = 1.0;
        OptimizationWithPreviousPoint();
      } else {
        // std::cout << "cluster begin" << camera_parameters_.begin()->second.at(0) << std::endl;
        OptimizationWithNesterovpoint(nesteorv_coeeficient);
        // std::cout << "cluster end" << camera_parameters_.begin()->second.at(0) << std::endl;
        last_cost_value = summary.final_cost;
     }
     {
       auto camera_boardcast = previous_external_camera_;
       auto point_boardcast = previous_external_point_;
       for (auto &[camera_id, parameters] : previous_external_camera_) {
         camera_boardcast[camera_id] = condition_external_camera_[camera_id];
       }

       for (auto &[point_id, parameters] : previous_external_point_) {
         point_boardcast[point_id] = condition_external_point_[point_id];
       }

       boardcast_callback_(iteration_, camera_boardcast, point_boardcast);
       ofs << cluster_id_ << " Send its parameters at " << iteration_ << std::endl;
     }
    }
    ofs.close();
  };
  thread_ = std::thread(solve_functor);
}
void DabaSubproblem::WaitForFinish() { thread_.join(); }

void DabaSubproblem::OptimizationWithPreviousPoint() {
  for (auto &[camera_id, parameters] : previous_internal_camera_) {
    std::copy(parameters.begin(), parameters.end(),
              camera_parameters_.at(camera_id).begin());
  }

  for (auto &[point_id, parameters] : previous_internal_point_) {
    std::copy(parameters.begin(), parameters.end(),
              point_parameters_.at(point_id).begin());
  }

  for (auto &[camera_id, parameters] : previous_external_camera_) {
    std::copy(parameters.begin(), parameters.end(),
              condition_external_camera_.at(camera_id).begin());
    std::copy(parameters.begin(), parameters.end(),
              camera_parameters_.at(camera_id).begin());
  }

  for (auto &[point_id, parameters] : previous_external_point_) {
    std::copy(parameters.begin(), parameters.end(),
              condition_external_point_.at(point_id).begin());
    std::copy(parameters.begin(), parameters.end(),
              point_parameters_.at(point_id).begin());
  }
}
void DabaSubproblem::OptimizationWithNesterovpoint(
    double nesteorv_coeeficient) {
  for (auto &[camera_id, parameters] : previous_external_camera_) {
    std::array<double, 9> optimized_camera_parameter =
        camera_parameters_.at(camera_id);
    std::array<double, 9> nesteorv_parameters = optimized_camera_parameter;
    NesteorvStep<double, 9>(parameters.data(), nesteorv_parameters.data(),
                            nesteorv_coeeficient);

    std::copy(optimized_camera_parameter.begin(),
              optimized_camera_parameter.end(), parameters.begin());

    std::copy(nesteorv_parameters.begin(), nesteorv_parameters.end(),
              condition_external_camera_.at(camera_id).begin());
  }

  for (auto &[point_id, parameters] : previous_external_point_) {
    auto optimized_point_parameter = point_parameters_.at(point_id);
    std::array<double, 3> nesteorv_parameters = optimized_point_parameter;

    NesteorvStep<double, 3>(parameters.data(), nesteorv_parameters.data(),
                            nesteorv_coeeficient);

    std::copy(optimized_point_parameter.begin(),
              optimized_point_parameter.end(), parameters.begin());

    std::copy(nesteorv_parameters.begin(), nesteorv_parameters.end(),
              condition_external_point_.at(point_id).begin());

    std::copy(nesteorv_parameters.begin(), nesteorv_parameters.end(),
              point_parameters_.at(point_id).begin());
  }

  for (auto &[camera_id, parameters] : previous_internal_camera_) {
    auto optimized_camera_parameter = camera_parameters_.at(camera_id);

    std::array<double, 9> nesteorv_parameters = optimized_camera_parameter;

    NesteorvStep<double, 9>(parameters.data(), nesteorv_parameters.data(),
                            nesteorv_coeeficient);
    std::copy(optimized_camera_parameter.begin(),
              optimized_camera_parameter.end(), parameters.begin());

    std::copy(nesteorv_parameters.begin(), nesteorv_parameters.end(),
              camera_parameters_.at(camera_id).begin());
  }

  for (auto &[point_id, parameters] : previous_internal_point_) {
    auto optimized_point_parameter = point_parameters_.at(point_id);

    std::array<double, 3> nesteorv_parameters = point_parameters_.at(point_id);

    NesteorvStep<double, 3>(parameters.data(), nesteorv_parameters.data(),
                            nesteorv_coeeficient);

    std::copy(optimized_point_parameter.begin(),
              optimized_point_parameter.end(), parameters.begin());

    std::copy(nesteorv_parameters.begin(), nesteorv_parameters.end(),
              point_parameters_.at(point_id).begin());
  }
}


void DabaSubproblem::CollectionNeighborParameters() {
  int64_t last_iteration = iteration_ - 1;
  if (last_iteration <= 0) {
    return;
  }

  std::map<int64_t, CameraParameters> last_iteration_neighbor_camera_parameters;
  std::map<int64_t, PointParameters> last_iteration_neighbor_point_parameters;
  
  int wait_count = 0;
  while (last_iteration_neighbor_camera_parameters.size() <
             external_other_camera_.size() ||
         last_iteration_neighbor_point_parameters.size() <
             external_other_point_.size()) {
    {
      wait_count++;
      std::lock_guard<std::mutex> lk_(external_queue_mutex_);
      auto temp_external_other_camera_queue = std::move(external_other_camera_queue_);
      auto temp_external_other_point_queue = std::move(external_other_point_queue_);
      while (!temp_external_other_camera_queue.empty()) {
        auto tuple = temp_external_other_camera_queue.front();
        int parameters_iteration = std::get<0>(tuple);
        int64_t camera_id = std::get<1>(tuple);
        auto camera_parameter = std::get<2>(tuple);
        temp_external_other_camera_queue.pop();

        if (parameters_iteration == last_iteration) {
          last_iteration_neighbor_camera_parameters[camera_id] = camera_parameter;
        } else if (parameters_iteration > last_iteration) {
          external_other_camera_queue_.push(tuple);
        }
      }
      while (!temp_external_other_point_queue.empty()) {
        auto tuple = temp_external_other_point_queue.front();
        int parameters_iteration = std::get<0>(tuple);
        int64_t point_id = std::get<1>(tuple);
        auto point_parameters = std::get<2>(tuple);
        temp_external_other_point_queue.pop();

        if (parameters_iteration == last_iteration) {
          last_iteration_neighbor_point_parameters[point_id] = point_parameters;
        } else if (parameters_iteration > last_iteration) {
          external_other_point_queue_.push(tuple);
        }
      }
    }
    // if (wait_count % 16 == 0) {
    //   std::cout << cluster_id_ << " Receive "
    //             << last_iteration_neighbor_camera_parameters.size()
    //             << " neighbor camera. Total " << external_other_camera_.size()
    //             << std::endl;
    //   std::cout << cluster_id_ << " Receive "
    //             << last_iteration_neighbor_point_parameters.size()
    //             << " neighbor point. Total " << external_other_point_.size()
    //             << std::endl;
    // }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // all the neighbor parameters received
  // update the parameters
  
  for (auto& [camera_id, external_neighbor_camera_parameters] : external_other_camera_) {
    assert(last_iteration_neighbor_camera_parameters.find(camera_id) != last_iteration_neighbor_camera_parameters.end());
    auto& parameters = last_iteration_neighbor_camera_parameters.at(camera_id);
    std::copy(parameters.begin(), parameters.end(), external_neighbor_camera_parameters.begin());
  }
  for (auto& [point_id, external_neighbor_point_parameters] : external_other_point_) {
    assert(last_iteration_neighbor_point_parameters.find(point_id) != last_iteration_neighbor_point_parameters.end());
    auto& parameters = last_iteration_neighbor_point_parameters.at(point_id);
    std::copy(parameters.begin(), parameters.end(), external_neighbor_point_parameters.begin());
  }
}

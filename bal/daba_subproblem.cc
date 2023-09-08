#include "daba_subproblem.h"

#include <ceres/autodiff_cost_function.h>

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
  camera_parameters_[camera_id] = camera_parameters;
  point_parameters_[point_id] = point_parameters;

  ceres::CostFunction *ray_cost_function =
      new ceres::AutoDiffCostFunction<RayCostFunction, 3, 9, 3>(
          new RayCostFunction(uv[0], uv[1]));

  problem_.AddResidualBlock(ray_cost_function, nullptr,
                            camera_parameters_[camera_id].data(),
                            point_parameters_[point_id].data());

  internal_costfunction_[{camera_id, point_id}] = ray_cost_function;
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
  camera_surrogate_costfunction_[camera_id] = camera_cost_function;
}

void DabaSubproblem::AddPoint(
    int64_t point_id, const std::array<double, 3> &point_parameters,
    int64_t external_camera_id,
    const std::array<double, 9> &external_camera_parameters,
    const std::array<double, 2> &uv) {
  point_parameters_[point_id] = point_parameters;
  previous_external_point_[point_id] = point_parameters;
  condition_external_point_[point_id] = point_parameters;

  external_other_camera_[external_camera_id] = external_camera_parameters;

  ceres::CostFunction *landmark_cost_function =
      new ceres::AutoDiffCostFunction<LandmarkSurrogatecostFunction, 3, 3>(
          new LandmarkSurrogatecostFunction(
              external_other_camera_.at(external_camera_id).data(),
              condition_external_point_.at(point_id).data(), uv[0], uv[1]));

  problem_.AddResidualBlock(landmark_cost_function, nullptr,
                            point_parameters_.at(point_id).data());
  point_surrogate_costfunction_[point_id] = landmark_cost_function;
}

// should be thread-safe
void DabaSubproblem::ReceiveExternalCamera(
    int64_t external_other_camera_id,
    const std::array<double, 9> &camera_parameters) {
  std::lock_guard<std::mutex> lk_(external_camera_queue_mutex_);
  external_other_camera_queue_.push(
      std::make_pair(external_other_camera_id, camera_parameters));
}

// should be thread-safe
void DabaSubproblem::ReceiveExternalPoint(
    int64_t external_other_point_id,
    const std::array<double, 3> &point_parameters) {
  std::lock_guard<std::mutex> lk_k(external_point_queue_mutex_);
  external_other_point_queue_.push(
      std::make_pair(external_other_point_id, point_parameters));
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
    constexpr int max_iteration = 16;
    int iteration = 0;

    double s = 1;
    double last_cost_value = std::numeric_limits<double>::max();

    while (iteration++ < max_iteration) {
      {
        std::lock_guard<std::mutex> lk_(external_camera_queue_mutex_);
        while (!external_other_camera_queue_.empty()) {
          auto pair = external_other_camera_queue_.front();
          external_other_camera_queue_.pop();
          std::copy(pair.second.begin(), pair.second.end(),
                    external_other_camera_[pair.first].begin());
        }
      }

      {
        std::lock_guard<std::mutex> lk_k(external_point_queue_mutex_);
        while (!external_other_point_queue_.empty()) {
          auto p = external_other_point_queue_.front();
          external_other_point_queue_.pop();
          std::copy(p.second.begin(), p.second.end(),
                    external_other_point_[p.first].begin());
        }
      }

      double s_next = (std::sqrt(4 * s * s + 1) + 1) * 0.5;
      double nesteorv_coeeficient = (s - 1) / s_next;
      s = s_next;

      ceres::Solver::Summary summary;
      ceres::Solver::Options options;
      options.max_num_iterations = 512;
      ceres::Solve(options, &problem_, &summary);
      std::cout << "cluster " << cluster_id_ << " iteration " << iteration
                << " summary : " << summary.BriefReport() << std::endl;
      if (summary.final_cost > last_cost_value) {
        std::cout << "cluster " << cluster_id_ << " restart." << iteration
                  << "cost increase from " << last_cost_value << " to "
                  << summary.final_cost << std::endl;
        s = 1.0;
        OptimizationWithPreviousPoint();
        for (auto &[camera_id, parameters] : previous_external_camera_) {
          boardcast_camera_callback_(camera_id,
                                     condition_external_camera_.at(camera_id));
        }

        for (auto &[point_id, parameters] : previous_external_point_) {
          boardcast_point_callback_(point_id,
                                    condition_external_point_.at(point_id));
        }
      } else {
        OptimizationWithNesterovpoint(nesteorv_coeeficient);
        for (auto &[camera_id, parameters] : previous_external_camera_) {
          boardcast_camera_callback_(camera_id,
                                     condition_external_camera_.at(camera_id));
        }

        for (auto &[point_id, parameters] : previous_external_point_) {
          boardcast_point_callback_(point_id,
                                    condition_external_point_.at(point_id));
        }
        last_cost_value = summary.final_cost;
      }
      
    }
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

#include "daba_subproblem.h"

#include <fstream>
#include <mutex>
#include <string>

#include "ceres/solver.h"
#include "cost_function_auto.h"

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

  // it should be update by ReceiveExternalPoint
  external_other_point_[external_point_id] = external_point_parameters;

  ceres::CostFunction *camera_cost_function =
      new ceres::AutoDiffCostFunction<CameraSurrogateCostFunction, 3, 9>(
          new CameraSurrogateCostFunction(
              previous_external_camera_[camera_id].data(),
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

  external_other_camera_[external_camera_id] = external_camera_parameters;

  ceres::CostFunction *landmark_cost_function =
      new ceres::AutoDiffCostFunction<LandmarkSurrogatecostFunction, 3, 3>(
          new LandmarkSurrogatecostFunction(
              external_other_camera_.at(external_camera_id).data(),
              previous_external_point_.at(point_id).data(), uv[0], uv[1]));

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
  auto solve_functor = [this]() {
    constexpr int max_iteration = 32;
    int iteration = 0;
    std::ofstream ofs(std::to_string(cluster_id_) + "loss.txt");

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

      // solve

      std::cout << "cluster " << cluster_id_ << " start solve" << std::endl;
      ceres::Solver::Summary summary;
      ceres::Solver::Options options;
      options.max_num_iterations = 16;
      ceres::Solve(options, &problem_, &summary);

      std::cout << "cluster " << cluster_id_ << " iteration " << iteration
                << " summary : " << summary.BriefReport() << std::endl;
      ofs << summary.BriefReport() << std::endl;

      for (auto &[camera_id, parameters] : previous_external_camera_) {
        std::copy(camera_parameters_.at(camera_id).begin(),
                  camera_parameters_.at(camera_id).end(), parameters.begin());
      }

      for (auto &[point_id, parameters] : previous_external_point_) {
        std::copy(point_parameters_.at(point_id).begin(),
                  point_parameters_.at(point_id).end(), parameters.begin());
      }

      for (auto &[camera_id, parameters] : previous_external_camera_) {
        boardcast_camera_callback_(camera_id, camera_parameters_.at(camera_id));
      }

      for (auto &[point_id, parameters] : previous_external_point_) {
        boardcast_point_callback_(point_id, point_parameters_.at(point_id));
      }
    }
    ofs.close();
  };
  thread_ = std::thread(solve_functor);
}
void DabaSubproblem::WaitForFinish() { thread_.join(); }
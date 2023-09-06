#ifndef BAL_DATA_SUBPROBLEM_H_
#define BAL_DATA_SUBPROBLEM_H_

#include <tuple>
#include <map>
#include <thread>
#include <queue>

#include "bal/problem.h"
#include "ceres/problem.h"

class DabaSubproblem {

public:
  DabaSubproblem(int cluster_id);

  void AddInternalEdge(int64_t camera_id,
                       const std::array<double, 9> &camera_parameters,
                       int64_t point_id,
                       const std::array<double, 3> &point_parameters,
                       std::array<double, 2> uv);

  void AddCamera(int64_t camera_id,
                 const std::array<double, 9> &camera_parameters,
                 int64_t external_point_id,
                 const std::array<double, 3> &external_point_parameters,
                 const std::array<double, 2> &uv);

  void AddPoint(int64_t point_id, const std::array<double, 3> &point_parameters,
                int64_t external_camera_id,
                const std::array<double, 9> &external_camera_parameters,
                const std::array<double, 2> &uv);

  // should be thread-safe
  void ReceiveExternalCamera(int64_t external_other_camera_id,
                             const std::array<double, 9> &camera_parameters);

  // should be thread-safe
  void ReceiveExternalPoint(int64_t external_other_point_id,
                            const std::array<double, 3> &point_parameters);
 
  using Edge = std::tuple<int64_t, int64_t, std::array<double, 2>>;
  using CameraParameters = std::array<double, 9>;
  using PointParameters = std::array<double, 3>;
  void SetBoardcastCallback(const std::function<void(int64_t, CameraParameters)>& boardcast_camera_callback) {
    boardcast_camera_callback_ = boardcast_camera_callback;
  }

  void SetBoardcastPointCallback(const std::function<void(int64_t, PointParameters)>& boardcast_point_callback) {
    boardcast_point_callback_ = boardcast_point_callback;
  }
  int ClusterId() const { return cluster_id_; }
  void Start();
  void WaitForFinish();

  std::map<int64_t, CameraParameters> ClusterCameraData() const {
    std::map<int64_t, CameraParameters>  res = internal_camera_;
    
    for (auto piar : external_camera_) {
        res.emplace(piar);
    }
    return res;
  }

  std::map<int64_t, PointParameters> ClusterPointData() const {
    std::map<int64_t, PointParameters> res = internal_point_;

    for (auto pair : external_point_) {
        res.emplace(pair);
    }
    return res;
  }

private:
  // camera id,  point id, uv

  std::map<int64_t, CameraParameters> internal_camera_;
  std::map<int64_t, PointParameters> internal_point_;

  std::map<int64_t, CameraParameters> external_camera_;
  std::map<int64_t, CameraParameters> previous_external_camera_;
  std::map<int64_t, PointParameters> external_point_;
  std::map<int64_t, PointParameters> previous_external_point_;

  std::map<int64_t, CameraParameters> external_other_camera_;
  std::map<int64_t, PointParameters> external_other_point_;
  std::queue<std::pair<int64_t, CameraParameters>> external_other_camera_queue_;
  std::mutex external_camera_queue_mutex_;
  std::queue<std::pair<int64_t, PointParameters>> external_other_point_queue_;
  std::mutex external_point_queue_mutex_;

  std::function<void(int64_t, CameraParameters)> boardcast_camera_callback_;
  std::function<void(int64_t, PointParameters)> boardcast_point_callback_;

  ceres::Problem problem_;
  std::thread thread_;
  int cluster_id_;

};

#endif //BAL_DATA_SUBPROBLEM_H_
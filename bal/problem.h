#ifndef BAL_PROBLEM_H_
#define BAL_PROBLEM_H_
#include <Eigen/Dense>
#include <map>
#include <unordered_map>
#include <vector>
#include <iostream>

typedef std::pair<size_t, size_t> IndexPair;

namespace std {
template <>
struct hash<IndexPair> {
  bool operator()(IndexPair const& rhs) const noexcept {
    std::size_t h1 = std::hash<size_t>{}(rhs.first);
    std::size_t h2 = std::hash<size_t>{}(rhs.second);
    return h1 & (h2 << 1);
  }
};
}  // namespace std
struct CameraParam {
  // R, t, f, k1, k2
  double params[9];
};

struct Landmark {
  double data[3];
  Landmark() {}
  Landmark(double x, double y, double z) {
    data[0] = x;
    data[1] = y;
    data[2] = z;
  }

  double operator()(size_t index) const {
    return data[index];
  }

  double& operator()(size_t index) {
    return data[index];
  }
};

struct Observation {
  double data[2];
  Observation() {}
  Observation(double x, double y) {
    data[0] = x;
    data[1] = y;
  }
  double operator()(size_t index) const {
    return data[index];
  }

  double& operator()(size_t index) {
    return data[index];
  }
};

struct Problem {
  std::unordered_map<size_t, CameraParam> cameras_;
  std::map<size_t, Landmark> points_;
  std::map<IndexPair, Observation> observations_;

  void Update(std::map<size_t, std::vector<double>> camera_step,
              std::map<size_t, std::vector<double>> point_step) {
    for (auto& camera_pair : camera_step) {
      for (size_t j = 0; j < 9; j++) {
        //std::cout << "camera_id : " << camera_pair.first << "delta " << j << " : " << camera_pair.second[j] << std::endl;
        //std::cout << "Before camera_id : " << camera_pair.first << "delta " << j << " : " << cameras_[camera_pair.first].params[j] << std::endl;
        cameras_[camera_pair.first].params[j] += camera_pair.second[j];
        //std::cout << "After camera_id : " << camera_pair.first << "delta " << j << " : " << cameras_[camera_pair.first].params[j] << std::endl;
      }
    }

    for (auto& point_pair : point_step) {
      for (size_t j = 0; j < 3; j++) {
        //std::cout << "Before point_id : " << point_pair.first << ": " << points_[point_pair.first](j) << std::endl;
        points_[point_pair.first](j) += point_pair.second[j];
        //std::cout << "After point_id : " << point_pair.first << ": " << points_[point_pair.first](j) << std::endl;
      }
    }
  }
};

#endif  // BAL_PROBLEM_H_
#ifndef BAL_PROBLEM_H_
#define BAL_PROBLEM_H_
#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>
#include "ceres/cost_function.h"
#include "cost_function_auto.h"

typedef std::pair<size_t, size_t> IndexPair;

namespace std {
template <> struct hash<IndexPair> {
  bool operator()(IndexPair const &rhs) const noexcept {
    std::size_t h1 = std::hash<size_t>{}(rhs.first);
    std::size_t h2 = std::hash<size_t>{}(rhs.second);
    return h1 & (h2 << 1);
  }
};
} // namespace std
struct CameraParam {
  // R, t, f, k1, k2
  double params[9];

  double *data() { return params; }

  const double *data() const { return params; }
};

struct Landmark {
  double data_[3];
  Landmark() {}
  Landmark(double x, double y, double z) : data_{x, y, z} {}

  double operator()(size_t index) const { return data_[index]; }

  double &operator()(size_t index) { return data_[index]; }

  double *data() { return data_; }

  const double *data() const { return data_; }
};

struct Observation {
  double data[2];

  Observation() {}

  Observation(double x, double y) {
    data[0] = x;
    data[1] = y;
  }

  double operator()(size_t index) const { return data[index]; }

  double &operator()(size_t index) { return data[index]; }

  double u() const { return data[0]; }

  double v() const { return data[1]; }
};

struct Problem {
  std::unordered_map<size_t, CameraParam> cameras_;
  std::map<size_t, Landmark> points_;
  std::map<IndexPair, Observation> observations_;

  void Update(std::map<size_t, std::vector<double>> camera_step,
              std::map<size_t, std::vector<double>> point_step) {
    for (auto &camera_pair : camera_step) {
      for (size_t j = 0; j < 9; j++) {
        cameras_[camera_pair.first].params[j] += camera_pair.second[j];
      }
    }

    for (auto &point_pair : point_step) {
      for (size_t j = 0; j < 3; j++) {
        points_[point_pair.first](j) += point_pair.second[j];
      }
    }
  }

  double MSE() const {
    double error = 0.0;
    for (auto &&[pairs, observation] : observations_) {
      const CameraParam &camera_parameter = cameras_.at(pairs.first);
      const double *camera_intrinsics = camera_parameter.data() + 6;
      const Landmark &points = points_.at(pairs.second);
      ceres::CostFunction *cost_func = ProjectFunction::CreateCostFunction(
          camera_intrinsics[0], camera_intrinsics[1], camera_intrinsics[2],
          observation.u(), observation.v());
      std::vector<const double*> parameters = {camera_parameter.data(), points.data()};
      double res[2];
      cost_func->Evaluate(parameters.data(), res, nullptr);

      error += res[0] * res[0] + res[1] * res[1];
    }
    return std::sqrt(error / observations_.size());
  }
};

#endif // BAL_PROBLEM_H_
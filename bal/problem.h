#ifndef BAL_PROBLEM_H_
#define BAL_PROBLEM_H_
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include "ceres/cost_function.h"
#include "cost_function_auto.h"

typedef std::pair<size_t, size_t> IndexPair;

struct CameraParam {
  // R_world_camera, t_world_camera, f, k1, k2
  double params[9];

  double *data() { return params; }

  const double *data() const { return params; }

  std::array<double, 9> array() const {
    std::array<double, 9> res;
    std::copy(params, params + 9, res.begin());
    return res;
  }

  static std::array<double, 9 + 3 + 3> ConvertLieAlgrebaToRotationMatrix(const std::array<double, 9>& params) {
    std::array<double, 9 + 3 + 3> res;

    ceres::AngleAxisToRotationMatrix(params.data(), res.data());
    res[9] = params[3];
    res[10] = params[4];
    res[11] = params[5];
    res[12] = params[6];
    res[13] = params[7];
    res[14] = params[8];
    return res;
  }
  
  static std::array<double, 9> Project(const std::array<double, 15>& parameters) {
    std::array<double, 9> res;

    Eigen::Matrix3d M = Eigen::Map<const Eigen::Matrix3d>(parameters.data());
    auto svd = M.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    Eigen::Matrix3d R = U * Eigen::Vector3d(1, 1, (U * V.transpose()).determinant()).asDiagonal() * V.transpose();
    ceres::RotationMatrixToAngleAxis(R.data(), res.data());
    res[3] = parameters[9];
    res[4] = parameters[10];
    res[5] = parameters[11];
    res[6] = parameters[12];
    res[7] = parameters[13];
    res[8] = parameters[14];
    return res;
  }

};

struct Landmark {
  double data_[3];
  Landmark() {}
  Landmark(double x, double y, double z) : data_{x, y, z} {}

  double operator()(size_t index) const { return data_[index]; }

  double &operator()(size_t index) { return data_[index]; }

  double *data() { return data_; }

  const double *data() const { return data_; }

  std::array<double, 3> array() const {
    std::array<double, 3> res;

    res[0] = data_[0];
    res[1] = data_[1];
    res[2] = data_[2];
    return res;
  }
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
    double ray_error = 0.0;
    for (auto &&[pairs, observation] : observations_) {
      const CameraParam &camera_parameter = cameras_.at(pairs.first);
      const Landmark &points = points_.at(pairs.second);
      ceres::CostFunction *cost_func = ProjectFunction::CreateCostFunction(
          observation.u(), observation.v());
      std::vector<const double*> parameters = {camera_parameter.data(), points.data()};
      double res[2];
      cost_func->Evaluate(parameters.data(), res, nullptr);
      RayCostFunction ray_cost_func(observation.u(), observation.v());
      double ray_res[3];
      ray_cost_func(camera_parameter.data(), points.data(), ray_res);
      error += res[0] * res[0] + res[1] * res[1];
      ray_error += ray_cost_func.EvaluateCost(camera_parameter.data(), points.data());
      Eigen::Vector2d res_map(res[0], res[1]);
      Eigen::Vector3d ray_res_map(ray_res[0], ray_res[1], ray_res[2]);

      //std::cout << "Project : norm : " << res_map.norm() << std::endl;
      //std::cout << "Res norm : " << ray_res_map.norm() << std::endl;
    }
    // return std::sqrt(error / observations_.size());
    std::cout << "ray_error : " << ray_error << std::endl;
    return ray_error / observations_.size();
  }

  bool ToPly(const std::string& ply_filename) {
    std::ofstream ofs(ply_filename);
    if (!ofs.is_open()) {
      return false;
    }
    
    ofs << "ply" << std::endl;
    ofs << "format ascii 1.0" << std::endl;
    ofs << "element vertex " << std::to_string(points_.size()) << std::endl;
    ofs << "property float x" << std::endl;
    ofs << "property float y" << std::endl;
    ofs << "property float z" << std::endl;
    ofs << "end_header" << std::endl;
    for (auto point : points_) {
      ofs << std::to_string(point.second.data_[0]) << " "
      << std::to_string(point.second.data_[1]) << " "
      << std::to_string(point.second.data_[2]) << std::endl;
    }
    ofs.close();
    return true;
  }
};

#endif // BAL_PROBLEM_H_
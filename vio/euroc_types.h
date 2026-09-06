#ifndef VIO_EUROC_TYPES_H_
#define VIO_EUROC_TYPES_H_
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "sophus/se3.hpp"

namespace vio {

struct CameraIntrinsics {
  double fu = 0, fv = 0, cu = 0, cv = 0;
};

struct RadTanDistortion {
  double k1 = 0, k2 = 0, p1 = 0, p2 = 0;
};

struct CameraCalibration {
  Sophus::SE3d T_BS;  // body_T_cam (camera-to-body extrinsic)
  CameraIntrinsics intrinsics;
  RadTanDistortion distortion;
  int width = 0, height = 0;
  double rate_hz = 0;
};

struct ImuCalibration {
  double rate_hz = 0;
  double gyro_noise_density = 0;   // rad/s / sqrt(Hz)
  double gyro_random_walk = 0;     // rad/s^2 / sqrt(Hz)
  double accel_noise_density = 0;  // m/s^2 / sqrt(Hz)
  double accel_random_walk = 0;    // m/s^3 / sqrt(Hz)
};

struct ImuSample {
  int64_t timestamp_ns = 0;
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();   // rad/s
  Eigen::Vector3d accel = Eigen::Vector3d::Zero();  // m/s^2
};

struct CameraFrameEntry {
  int64_t timestamp_ns = 0;
  std::string filename;
};

struct GroundTruthSample {
  int64_t timestamp_ns = 0;
  Eigen::Vector3d p_world = Eigen::Vector3d::Zero();
  Sophus::SO3d R_world_body;  // R_RS in EuRoC's notation
  Eigen::Vector3d v_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d bias_gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d bias_accel = Eigen::Vector3d::Zero();
};

struct EurocSequence {
  std::string mav0_dir;
  CameraCalibration cam0, cam1;
  ImuCalibration imu0;
  std::vector<ImuSample> imu_samples;          // sorted by timestamp_ns
  std::vector<CameraFrameEntry> cam0_frames;   // sorted by timestamp_ns
  std::vector<CameraFrameEntry> cam1_frames;   // sorted by timestamp_ns
  std::vector<GroundTruthSample> ground_truth; // sorted by timestamp_ns
};

}  // namespace vio
#endif  // VIO_EUROC_TYPES_H_

#ifndef VIO_VIO_SIMULATOR_H_
#define VIO_VIO_SIMULATOR_H_
#include <cstdint>
#include <vector>

#include <Eigen/Dense>

#include "eqvio.h"
#include "imu_trajectory_simulator.h"
#include "sophus/se3.hpp"

namespace vio {

// Synthetic monocular VIO scene following the bias/extrinsic-convergence
// experiment of van Goor & Mahony, EqVIO (arXiv:2205.01980), Sec. 7.3:
//   R_B(t) = Exp(pi/4 * (cos 0.25t, cos(-0.3t), cos 0.2t))
//   x_B(t) = 1/2 * (cos 0.1 pi t, cos 0.2 pi t, cos 0.15 pi t)
// with landmarks scattered uniformly on the six faces of a cube one metre
// outside the trajectory's bounds. The rotation about all three axes is what
// makes the extrinsics and both biases observable.
struct VioSimConfig {
  double duration_s = 30.0;
  double imu_hz = 200.0;
  double camera_hz = 20.0;  // must divide imu_hz

  ImuNoiseParams noise;
  bool add_imu_noise = true;
  bool add_bias_random_walk = false;
  Eigen::Vector3d initial_bias_gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d initial_bias_accel = Eigen::Vector3d::Zero();
  Eigen::Vector3d gravity_world = Eigen::Vector3d(0, 0, -9.81);

  double box_half_extent_m = 1.5;  // trajectory spans +-0.5 m, plus 1 m
  int landmarks_per_face = 50;

  // Pinhole camera, EuRoC cam0-sized by default.
  double focal_px = 458.0;
  int width_px = 752;
  int height_px = 480;
  double pixel_sigma = 1.0;
  double min_depth_m = 0.1;
  int max_tracked_features = 30;

  Sophus::SE3d T_body_camera =
      Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(0.1, -0.05, 0.02)),
                   Eigen::Vector3d(0.05, 0.0, 0.02));

  std::uint32_t seed = 7;
};

struct CameraFrame {
  double t = 0.0;
  std::size_t imu_index = 0;  // index into VioSimulation::imu at time t
  std::vector<BearingMeasurement> bearings;
};

struct VioSimulation {
  std::vector<Eigen::Vector3d> landmarks_world;  // id == index
  Sophus::SE3d T_body_camera;
  std::vector<ImuTrajectorySample> imu;
  std::vector<CameraFrame> frames;
};

VioSimulation SimulateVio(const VioSimConfig& config);

// Ground truth at imu[k] as an EqvioState holding every landmark, so it can
// be passed to EqvioEstimator::ErrorCoordinates.
EqvioState TruthState(const VioSimulation& sim, std::size_t k);

}  // namespace vio
#endif  // VIO_VIO_SIMULATOR_H_

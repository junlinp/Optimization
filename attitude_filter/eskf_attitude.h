#ifndef ATTITUDE_FILTER_ESKF_ATTITUDE_H_
#define ATTITUDE_FILTER_ESKF_ATTITUDE_H_
#include <Eigen/Dense>

#include "sophus/so3.hpp"

namespace attitude_filter {

// A minimal attitude-only ESKF: orientation + gyro bias, updated from raw
// gyro readings and from body-frame observations of known world-frame
// reference vectors (gravity, local magnetic field). No position/velocity --
// see vio::EskfEstimator for the full 15-state VIO filter this is a stripped
// down cousin of.
//
// Nominal state. Error state (used only internally, for propagation/update)
// is 6-dim, ordered [dtheta(0:3), db_g(3:6)], related to this by:
// R_true=R*Exp(dtheta), b_g_true=b_g+db_g -- same convention as
// vio::EskfState.
struct AttitudeEskfState {
  Sophus::SO3d R;  // R_world_body, defaults to identity
  Eigen::Vector3d bias_gyro = Eigen::Vector3d::Zero();
};

struct AttitudeImuNoiseParams {
  double gyro_noise_density = 0;  // rad/s / sqrt(Hz)
  double gyro_random_walk = 0;    // rad/s^2 / sqrt(Hz)
};

// A single body-frame observation of a vector whose direction is known in
// the world frame (e.g. gravity, or the local magnetic field). Both vectors
// are expected normalized; sigma is the per-axis measurement-noise std of
// the small-angle residual (documented simplification: the true measurement
// covariance is rank-2 on the sphere, but a diagonal 3D Gaussian is treated
// as a practical approximation, same spirit as RelativePoseMeasurement's
// fixed sigmas in vio::EskfEstimator).
struct VectorMeasurement {
  Eigen::Vector3d body_measured = Eigen::Vector3d::Zero();
  Eigen::Vector3d world_reference = Eigen::Vector3d::Zero();
  double sigma = 0.01;
};

class AttitudeEskf {
 public:
  AttitudeEskf(const AttitudeEskfState& initial_state,
              const Eigen::Matrix<double, 6, 6>& initial_covariance,
              const AttitudeImuNoiseParams& noise_params);

  // Strapdown rotation propagation + error-state covariance propagation.
  // gyro_meas is a raw (bias- and noise-corrupted) reading; dt is the time
  // since the previous Predict call, in seconds, dt > 0.
  void Predict(const Eigen::Vector3d& gyro_meas, double dt);

  // Applies a vector-observation EKF correction (e.g. gravity or
  // magnetometer). Call once per available sensor per timestep.
  void UpdateVector(const VectorMeasurement& measurement);

  const AttitudeEskfState& state() const { return state_; }
  const Eigen::Matrix<double, 6, 6>& covariance() const { return P_; }

 private:
  AttitudeEskfState state_;
  Eigen::Matrix<double, 6, 6> P_;
  AttitudeImuNoiseParams noise_;
};

}  // namespace attitude_filter
#endif  // ATTITUDE_FILTER_ESKF_ATTITUDE_H_

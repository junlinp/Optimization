#ifndef VIO_ESKF_ESTIMATOR_H_
#define VIO_ESKF_ESTIMATOR_H_
#include <Eigen/Dense>

#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

namespace vio {

// Nominal state. Error state (used only internally, for propagation/update)
// is 15-dim, ordered [dp(0:3), dv(3:6), dtheta(6:9), db_g(9:12), db_a(12:15)],
// related to this by: p_true=p+dp, v_true=v+dv, R_true=R*Exp(dtheta),
// b_g_true=b_g+db_g, b_a_true=b_a+db_a.
struct EskfState {
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  Eigen::Vector3d v = Eigen::Vector3d::Zero();
  Sophus::SO3d R;  // R_world_body, defaults to identity
  Eigen::Vector3d bias_gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d bias_accel = Eigen::Vector3d::Zero();
};

struct ImuNoiseParams {
  double gyro_noise_density = 0;   // rad/s / sqrt(Hz)
  double gyro_random_walk = 0;     // rad/s^2 / sqrt(Hz)
  double accel_noise_density = 0;  // m/s^2 / sqrt(Hz)
  double accel_random_walk = 0;    // m/s^3 / sqrt(Hz)
};

// z: VO-measured relative body pose T_prevbody_currbody (maps points
// expressed in current-frame body coordinates into previous-frame body
// coordinates). sigma_* are fixed default measurement-noise stds -- not
// derived from RANSAC inlier statistics (documented limitation).
struct RelativePoseMeasurement {
  Sophus::SE3d T_prevbody_currbody;
  double sigma_translation_m = 0.02;
  double sigma_rotation_rad = 0.01;
};

class EskfEstimator {
 public:
  EskfEstimator(const EskfState& initial_state,
                const Eigen::Matrix<double, 15, 15>& initial_covariance,
                const ImuNoiseParams& noise_params,
                const Eigen::Vector3d& gravity_world = Eigen::Vector3d(0, 0, -9.81));

  // Strapdown IMU mechanization + error-state covariance propagation.
  // gyro_meas/accel_meas are raw (bias- and noise-corrupted) readings; dt is
  // the time since the previous Predict call, in seconds, dt > 0.
  void Predict(const Eigen::Vector3d& gyro_meas, const Eigen::Vector3d& accel_meas, double dt);

  // Anchors the current nominal pose as T_k for the NEXT UpdateRelativePose
  // call. Call once per camera frame, after any update at that frame (or
  // immediately, on the very first frame that has no prior anchor).
  void SetRelativePoseAnchor();

  // Applies the relative-pose EKF correction. Returns false (no state/
  // covariance change) if SetRelativePoseAnchor() was never called.
  bool UpdateRelativePose(const RelativePoseMeasurement& measurement);

  const EskfState& state() const { return state_; }
  const Eigen::Matrix<double, 15, 15>& covariance() const { return P_; }

 private:
  EskfState state_;
  Eigen::Matrix<double, 15, 15> P_;
  ImuNoiseParams noise_;
  Eigen::Vector3d gravity_world_;
  bool have_anchor_ = false;
  Sophus::SO3d anchor_R_;
  Eigen::Vector3d anchor_p_ = Eigen::Vector3d::Zero();
};

}  // namespace vio
#endif  // VIO_ESKF_ESTIMATOR_H_

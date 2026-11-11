#ifndef VIO_EQVIO_H_
#define VIO_EQVIO_H_
// EqVIO: equivariant filter for visual-inertial odometry, after
// P. van Goor and R. Mahony, "EqVIO: An Equivariant Filter for Visual
// Inertial Odometry", IEEE T-RO 2023 (arXiv:2205.01980).
//
// State space T^n_VI(3) (Sec. 4): IMU pose P = (R, x), velocity v, biases b,
// camera extrinsic T (camera pose in the body frame), and n landmarks. The
// landmarks are kept robocentrically, as q_i = (P T)^-1 p_i in the camera
// frame, because that is the coordinate both the SOT(3) symmetry and the
// bearing measurement act on.
//
// Symmetry group (eq. 22): G = SE_2(3) x R^6 x SE(3) x SOT(3)^n with the
// right action (eq. 23)
//   phi(X, xi) = (N A, b + beta, P_A^-1 T B, c_i^-1 R_i^T q_i),
// where N = (R, x, v) is the navigation state viewed as an SE_2(3) element.
//
// Origin xi0: N = I, b = 0, T = I, q_i = e3. Taking T0 = I (the paper leaves
// it free) makes the observer's B component equal to the camera pose
// estimate P T. The filter state is the group element X_hat; the estimate is
// xi_hat = phi(X_hat, xi0).
//
// Local error coordinates eps = vartheta(phi(X_hat^-1, xi)) (eq. 27), with
// dimension 21 + 3n, ordered
//   [ 0: 9) log_SE23(N N_hat^-1)            right-invariant navigation error
//   [ 9:15) b - b_hat                         gyro bias then accel bias
//   [15:21) log_SE3(C C_hat^-1), C = P T      camera-pose error, Sophus order (upsilon, omega)
//   [21+3i) polar(c_i R_i q_i)                landmark i, normal coords of SOT(3) about e3
// The Riccati matrix Sigma is the covariance of eps.
#include <vector>

#include <Eigen/Dense>

#include "eqvio_lie_groups.h"
#include "eskf_estimator.h"
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

namespace vio {

using Vector6d = Eigen::Matrix<double, 6, 1>;

constexpr int kEqvioCoreDim = 21;  // navigation(9) + bias(6) + camera(6)

struct EqvioState {
  Sophus::SO3d R;  // R_world_body
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  Eigen::Vector3d v = Eigen::Vector3d::Zero();
  Eigen::Vector3d bias_gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d bias_accel = Eigen::Vector3d::Zero();
  Sophus::SE3d T_body_camera;

  // Parallel arrays. landmarks_camera[i] = (P T)^-1 p_i for id landmark_ids[i].
  std::vector<int> landmark_ids;
  std::vector<Eigen::Vector3d> landmarks_camera;

  Sophus::SE3d T_world_body() const { return Sophus::SE3d(R, p); }
  Sophus::SE3d T_world_camera() const { return T_world_body() * T_body_camera; }
  Eigen::Vector3d LandmarkWorld(std::size_t i) const {
    return T_world_camera() * landmarks_camera[i];
  }
};

// X = (A, beta, B, Q_1..Q_n).
struct EqvioGroup {
  SE23 A;
  Vector6d beta = Vector6d::Zero();
  Sophus::SE3d B;
  std::vector<SOT3> Q;

  EqvioGroup operator*(const EqvioGroup& other) const;
  EqvioGroup inverse() const;
};

// An element of the Lie algebra g of G.
struct EqvioAlgebra {
  Vector9d A = Vector9d::Zero();
  Vector6d beta = Vector6d::Zero();
  Vector6d B = Vector6d::Zero();       // Sophus order (upsilon, omega)
  std::vector<Eigen::Vector4d> Q;      // (omega, s)
};

struct BearingMeasurement {
  int id = 0;
  Eigen::Vector3d bearing = Eigen::Vector3d::UnitZ();  // unit vector, camera frame
};

// The pieces of EqVIO's geometry, exposed so tests can check them
// independently of the filter.
namespace eqvio {

EqvioState Origin(const std::vector<int>& landmark_ids);

// phi(X, xi), eq. (23). X.Q.size() must equal xi.landmarks_camera.size().
EqvioState Act(const EqvioGroup& X, const EqvioState& xi);

// Polar parametrisation (eq. 16) in this file's sign convention: the normal
// coordinates z of q about e3 for the action phi_V, i.e.
// q = phi_V(exp((z0, z1, 0), z2), e3) = e^-z2 Exp(-(z0, z1, 0)) e3.
Eigen::Vector3d PolarCoordinates(const Eigen::Vector3d& q);
Eigen::Vector3d FromPolarCoordinates(const Eigen::Vector3d& z);

// vartheta, eq. (27), about Origin(), and its inverse.
Eigen::VectorXd LocalCoordinates(const EqvioState& e);
EqvioState FromLocalCoordinates(const Eigen::VectorXd& eps, const std::vector<int>& landmark_ids);

// The lift Lambda(xi, u) of Lemma 6.1, with the IMU biases and gravity
// written out (the paper's eq. 25 elides both): the SE_2(3) part is
// (w, R^T v, a + R^T g) with w = gyro - b_g, a = accel - b_a.
EqvioAlgebra Lift(const EqvioState& xi, const Eigen::Vector3d& gyro_meas,
                  const Eigen::Vector3d& accel_meas, const Eigen::Vector3d& gravity_world);

EqvioGroup Exp(const EqvioAlgebra& lambda, double dt);

// Right inverse of D phi_xi0 composed with vartheta^-1: the group element Z
// with vartheta(phi(Z, xi0)) == eps exactly (vartheta is normal for phi).
// The SOT(3) rotation about e3 (stabiliser of xi0) is set to zero.
EqvioGroup CorrectionFromLocalCoordinates(const Eigen::VectorXd& eps);

// Builds the 21x21 covariance of eps[0:21) from independent priors on the
// navigation error (9x9, in eps's own right-invariant SE_2(3) coordinates),
// the biases (6x6) and the extrinsic (6x6, a right perturbation
// T = T_hat Exp(tau) in Sophus (upsilon, omega) order). eps_C is the
// camera-pose error, so it inherits the body-pose error too:
// eps_C ~= (eps_x, eps_R) + Ad_{C_hat} tau.
Eigen::MatrixXd InitialCovariance(const EqvioState& estimate,
                                  const Eigen::Matrix<double, 9, 9>& navigation,
                                  const Eigen::Matrix<double, 6, 6>& bias,
                                  const Eigen::Matrix<double, 6, 6>& extrinsic);

}  // namespace eqvio

struct EqvioParams {
  ImuNoiseParams imu;
  Eigen::Vector3d gravity_world = Eigen::Vector3d(0, 0, -9.81);

  double bearing_sigma = 1e-3;  // rad, std of the measured unit bearing

  // A new landmark is initialised on its measured bearing at the median
  // depth of the landmarks already in the state, or at this depth when there
  // are none, with covariance diag(sigma_b^2, sigma_b^2, sigma_logdepth^2).
  double initial_landmark_depth = 2.0;
  double initial_landmark_bearing_sigma = 1e-2;
  double initial_landmark_log_depth_sigma = 0.7;

  // State gain M_eps (per second) on the camera and landmark blocks.
  double camera_process_noise = 0.0;
  double landmark_process_noise = 0.0;

  // Per-landmark chi-square gate on r^T S^-1 r (2 dof). <= 0 disables.
  double outlier_chi2_threshold = 0.0;
};

class EqvioEstimator {
 public:
  struct UpdateResult {
    int num_used = 0;
    int num_added = 0;
    int num_removed = 0;   // tracked landmarks with no measurement this frame
    int num_outliers = 0;  // rejected by the chi-square gate (also removed)
  };

  // initial_state's landmarks are ignored; landmarks enter through Update().
  // initial_covariance is the 21x21 covariance of eps[0:21) (see above).
  EqvioEstimator(const EqvioState& initial_state, const Eigen::MatrixXd& initial_covariance,
                 const EqvioParams& params);

  // EqF propagation (eq. 28 without the correction): X <- X exp(Lambda dt),
  // Sigma <- Phi Sigma Phi^T + (B M B^T + M_eps) dt with Phi = I + A dt.
  void Predict(const Eigen::Vector3d& gyro_meas, const Eigen::Vector3d& accel_meas, double dt);

  // One camera frame. Landmarks absent from `measurements` are removed, new
  // ids are added, then the EqF correction is applied with the equivariant
  // output matrix C*.
  UpdateResult Update(const std::vector<BearingMeasurement>& measurements);

  EqvioState state() const;
  const EqvioGroup& observer() const { return X_; }
  const Eigen::MatrixXd& covariance() const { return Sigma_; }
  const std::vector<int>& landmark_ids() const { return ids_; }

  // eps = vartheta(phi(X_hat^-1, truth)). truth must contain every tracked
  // landmark id; they are looked up by id and extras are ignored.
  Eigen::VectorXd ErrorCoordinates(const EqvioState& truth) const;

  // Continuous-time EqF state matrix A0_t (dim x dim) and input matrix B_t
  // (dim x 6, columns gyro noise then accel noise) at the current estimate.
  // Derived by hand from the error dynamics; see eqvio.cc.
  Eigen::MatrixXd StateMatrix(const Eigen::Vector3d& gyro_meas,
                              const Eigen::Vector3d& accel_meas) const;
  Eigen::MatrixXd InputMatrix(const Eigen::Vector3d& gyro_meas,
                              const Eigen::Vector3d& accel_meas) const;

  // The equivariant output matrix C*_t (eq. 28) for landmark slot i and
  // measured bearing y, 2x3, acting on that landmark's eps block; and the
  // matching residual. Both are projected onto the e1/e2 plane.
  static Eigen::Matrix<double, 2, 3> OutputMatrix(const Eigen::Vector3d& y_origin);
  Eigen::Vector3d BearingAtOrigin(std::size_t slot, const Eigen::Vector3d& bearing) const;

 private:
  struct Linearisation {
    Eigen::Matrix<double, kEqvioCoreDim, kEqvioCoreDim> F;
    Eigen::Matrix<double, kEqvioCoreDim, 6> B;
    std::vector<Eigen::Matrix<double, 3, kEqvioCoreDim>> L;  // landmark rows, core columns
    std::vector<Eigen::Matrix3d> D;                          // landmark diagonal blocks
    std::vector<Eigen::Matrix<double, 3, 6>> BL;             // landmark rows of B
  };
  Linearisation Linearise(const Eigen::Vector3d& gyro_meas,
                          const Eigen::Vector3d& accel_meas) const;

  void AddLandmark(int id, const Eigen::Vector3d& bearing, double depth);
  void RemoveLandmarks(const std::vector<bool>& keep);

  EqvioParams params_;
  EqvioGroup X_;
  Eigen::MatrixXd Sigma_;
  std::vector<int> ids_;
};

}  // namespace vio
#endif  // VIO_EQVIO_H_

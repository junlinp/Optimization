#include "eqvio.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace vio {
namespace {

using Matrix3d = Eigen::Matrix3d;
using Vector3d = Eigen::Vector3d;

Matrix3d Hat(const Vector3d& w) { return Sophus::SO3d::hat(w); }

// d(q^e)/d(eps_q) at q^e = e3, from FromPolarCoordinates:
// e^-z2 Exp(-(z0,z1,0)) e3 ~= e3 + (-z1, z0, -z2). Orthogonal, so M^-1 = M^T.
Matrix3d PolarJacobianAtE3() {
  Matrix3d M;
  M << 0, -1, 0,
       1, 0, 0,
       0, 0, -1;
  return M;
}

int LandmarkOffset(std::size_t i) { return kEqvioCoreDim + 3 * static_cast<int>(i); }

}  // namespace

EqvioGroup EqvioGroup::operator*(const EqvioGroup& other) const {
  EqvioGroup out;
  out.A = A * other.A;
  out.beta = beta + other.beta;
  out.B = B * other.B;
  out.Q.resize(Q.size());
  for (std::size_t i = 0; i < Q.size(); ++i) out.Q[i] = Q[i] * other.Q[i];
  return out;
}

EqvioGroup EqvioGroup::inverse() const {
  EqvioGroup out;
  out.A = A.inverse();
  out.beta = -beta;
  out.B = B.inverse();
  out.Q.resize(Q.size());
  for (std::size_t i = 0; i < Q.size(); ++i) out.Q[i] = Q[i].inverse();
  return out;
}

namespace eqvio {

EqvioState Origin(const std::vector<int>& landmark_ids) {
  EqvioState xi;
  xi.landmark_ids = landmark_ids;
  xi.landmarks_camera.assign(landmark_ids.size(), Vector3d::UnitZ());
  return xi;
}

EqvioState Act(const EqvioGroup& X, const EqvioState& xi) {
  if (X.Q.size() != xi.landmarks_camera.size()) {
    throw std::invalid_argument("group and state have different landmark counts");
  }
  EqvioState out;
  const SE23 N = SE23{xi.R, xi.p, xi.v} * X.A;
  out.R = N.R;
  out.p = N.x;
  out.v = N.v;
  out.bias_gyro = xi.bias_gyro + X.beta.head<3>();
  out.bias_accel = xi.bias_accel + X.beta.tail<3>();
  out.T_body_camera = X.A.pose().inverse() * xi.T_body_camera * X.B;
  out.landmark_ids = xi.landmark_ids;
  out.landmarks_camera.resize(xi.landmarks_camera.size());
  for (std::size_t i = 0; i < xi.landmarks_camera.size(); ++i) {
    out.landmarks_camera[i] = X.Q[i].Act(xi.landmarks_camera[i]);
  }
  return out;
}

Vector3d PolarCoordinates(const Vector3d& q) {
  const double r = q.norm();
  const Vector3d y = q / r;
  // Exp(-omega) e3 = y: -omega is the minimal rotation taking e3 to y, whose
  // axis e3 x y lies in the xy-plane, so omega_z = 0 automatically.
  const Vector3d k = Vector3d::UnitZ().cross(y);
  const double sin_theta = k.norm();
  const Vector3d omega =
      sin_theta > 1e-12 ? Vector3d(-std::atan2(sin_theta, y.z()) * k / sin_theta) : Vector3d(-k);
  return {omega.x(), omega.y(), -std::log(r)};
}

Vector3d FromPolarCoordinates(const Vector3d& z) {
  return std::exp(-z.z()) * (Sophus::SO3d::exp(-Vector3d(z.x(), z.y(), 0)) * Vector3d::UnitZ());
}

Eigen::VectorXd LocalCoordinates(const EqvioState& e) {
  const std::size_t n = e.landmarks_camera.size();
  Eigen::VectorXd eps(kEqvioCoreDim + 3 * n);
  eps.segment<9>(0) = SE23::log(SE23{e.R, e.p, e.v});
  eps.segment<3>(9) = e.bias_gyro;
  eps.segment<3>(12) = e.bias_accel;
  eps.segment<6>(15) = e.T_world_camera().log();
  for (std::size_t i = 0; i < n; ++i) {
    eps.segment<3>(LandmarkOffset(i)) = PolarCoordinates(e.landmarks_camera[i]);
  }
  return eps;
}

EqvioState FromLocalCoordinates(const Eigen::VectorXd& eps, const std::vector<int>& landmark_ids) {
  EqvioState e;
  const SE23 N = SE23::exp(eps.segment<9>(0));
  e.R = N.R;
  e.p = N.x;
  e.v = N.v;
  e.bias_gyro = eps.segment<3>(9);
  e.bias_accel = eps.segment<3>(12);
  e.T_body_camera = e.T_world_body().inverse() * Sophus::SE3d::exp(eps.segment<6>(15));
  e.landmark_ids = landmark_ids;
  for (std::size_t i = 0; i < landmark_ids.size(); ++i) {
    e.landmarks_camera.push_back(FromPolarCoordinates(eps.segment<3>(LandmarkOffset(i))));
  }
  return e;
}

EqvioAlgebra Lift(const EqvioState& xi, const Vector3d& gyro_meas, const Vector3d& accel_meas,
                  const Vector3d& gravity_world) {
  EqvioAlgebra lambda;
  const Vector3d omega = gyro_meas - xi.bias_gyro;
  const Vector3d a = accel_meas - xi.bias_accel;
  const Vector3d v_body = xi.R.inverse() * xi.v;
  lambda.A << omega, v_body, a + xi.R.inverse() * gravity_world;

  // B = Ad_{T^-1} U keeps T = P_A^-1 T B constant; (upsilon, omega) is then
  // the camera's own body-frame twist.
  Vector6d U;
  U << v_body, omega;
  lambda.B = xi.T_body_camera.inverse().Adj() * U;
  const Vector3d v_cam = lambda.B.head<3>();
  const Vector3d omega_cam = lambda.B.tail<3>();

  // Landmark i moves as q_dot = -Omega_C x q - v_C; split v_C into a part
  // along q (absorbed by the SOT(3) scale) and a part normal to it (absorbed
  // by an extra rotation).
  for (const Vector3d& q : xi.landmarks_camera) {
    const double n2 = q.squaredNorm();
    Eigen::Vector4d lq;
    lq << omega_cam + q.cross(v_cam) / n2, q.dot(v_cam) / n2;
    lambda.Q.push_back(lq);
  }
  return lambda;
}

EqvioGroup Exp(const EqvioAlgebra& lambda, double dt) {
  EqvioGroup X;
  X.A = SE23::exp(lambda.A * dt);
  X.beta = lambda.beta * dt;
  X.B = Sophus::SE3d::exp(lambda.B * dt);
  for (const Eigen::Vector4d& lq : lambda.Q) {
    X.Q.push_back(SOT3::exp(lq.head<3>() * dt, lq(3) * dt));
  }
  return X;
}

EqvioGroup CorrectionFromLocalCoordinates(const Eigen::VectorXd& eps) {
  EqvioGroup Z;
  Z.A = SE23::exp(eps.segment<9>(0));
  Z.beta = eps.segment<6>(9);
  Z.B = Sophus::SE3d::exp(eps.segment<6>(15));
  const int n = (static_cast<int>(eps.size()) - kEqvioCoreDim) / 3;
  for (int i = 0; i < n; ++i) {
    const Vector3d z = eps.segment<3>(LandmarkOffset(i));
    Z.Q.push_back(SOT3::exp(Vector3d(z.x(), z.y(), 0), z.z()));
  }
  return Z;
}

Eigen::MatrixXd InitialCovariance(const EqvioState& estimate,
                                  const Eigen::Matrix<double, 9, 9>& navigation,
                                  const Eigen::Matrix<double, 6, 6>& bias,
                                  const Eigen::Matrix<double, 6, 6>& extrinsic) {
  // eps_core = J * (eps_nav, eps_bias, tau) with independent priors.
  Eigen::Matrix<double, kEqvioCoreDim, kEqvioCoreDim> J =
      Eigen::Matrix<double, kEqvioCoreDim, kEqvioCoreDim>::Identity();
  J.block<3, 3>(15, 15).setZero();
  J.block<3, 3>(18, 18).setZero();
  J.block<3, 3>(15, 3) = Matrix3d::Identity();  // upsilon <- eps_x
  J.block<3, 3>(18, 0) = Matrix3d::Identity();  // omega <- eps_R
  J.block<6, 6>(15, 15) = estimate.T_world_camera().Adj();

  Eigen::Matrix<double, kEqvioCoreDim, kEqvioCoreDim> prior =
      Eigen::Matrix<double, kEqvioCoreDim, kEqvioCoreDim>::Zero();
  prior.block<9, 9>(0, 0) = navigation;
  prior.block<6, 6>(9, 9) = bias;
  prior.block<6, 6>(15, 15) = extrinsic;
  return J * prior * J.transpose();
}

}  // namespace eqvio

EqvioEstimator::EqvioEstimator(const EqvioState& initial_state,
                               const Eigen::MatrixXd& initial_covariance,
                               const EqvioParams& params)
    : params_(params), Sigma_(initial_covariance) {
  if (initial_covariance.rows() != kEqvioCoreDim || initial_covariance.cols() != kEqvioCoreDim) {
    throw std::invalid_argument("initial covariance must be 21x21");
  }
  // xi_hat = phi(X, xi0) with xi0's N = I, b = 0, T = I gives
  // A = N_hat, beta = b_hat, B = P_hat T_hat.
  X_.A = SE23{initial_state.R, initial_state.p, initial_state.v};
  X_.beta << initial_state.bias_gyro, initial_state.bias_accel;
  X_.B = initial_state.T_world_camera();
}

EqvioState EqvioEstimator::state() const { return eqvio::Act(X_, eqvio::Origin(ids_)); }

Eigen::VectorXd EqvioEstimator::ErrorCoordinates(const EqvioState& truth) const {
  std::unordered_map<int, std::size_t> slot;
  for (std::size_t i = 0; i < truth.landmark_ids.size(); ++i) slot[truth.landmark_ids[i]] = i;
  EqvioState sub = truth;
  sub.landmark_ids = ids_;
  sub.landmarks_camera.clear();
  for (const int id : ids_) {
    const auto it = slot.find(id);
    if (it == slot.end()) throw std::invalid_argument("truth is missing a tracked landmark");
    sub.landmarks_camera.push_back(truth.landmarks_camera[it->second]);
  }
  return eqvio::LocalCoordinates(eqvio::Act(X_.inverse(), sub));
}

// Linearised error dynamics eps_dot = A0 eps + B n at the current estimate.
// Writing the true state as xi = phi(X_hat, vartheta^-1(eps)) and
// differentiating eps = vartheta(phi(X_hat^-1, xi)) along both the true
// dynamics (inputs u - n) and the observer's lifted dynamics gives, to first
// order (with w = R_hat (gyro - b_g_hat) the world-frame angular velocity):
//
//   navigation (right-invariant SE_2(3) error with biases):
//     d eps_R = -R db_g                         - R n_g
//     d eps_x = eps_v - x^ R db_g               - x^ R n_g
//     d eps_v = g^ eps_R - v^ R db_g - R db_a   - v^ R n_g - R n_a
//   camera pose error E = C C_hat^-1, with a = Ad_P U the body's (equally
//   the camera's) spatial twist:
//     d eps_C = delta_a + ad_{a_hat} eps_C
//   landmark i, with q^e = c_i R_i q:
//     d eps_q = M^T c R [ Psi (cR)^-1 M eps_q + [-I, q^] Ad_{C_hat^-1} d eps_C ],
//     Psi = s I + (q x v_C / |q|^2)^, s = q.v_C / |q|^2.
// The landmark term uses that the camera's body twist is perturbed by
// exactly Ad_{C_hat^-1} d eps_C. EqvioTest.StateMatrixMatchesNumerical checks
// all of this against finite differences of the true error dynamics.
EqvioEstimator::Linearisation EqvioEstimator::Linearise(const Vector3d& gyro_meas,
                                                        const Vector3d& accel_meas) const {
  const EqvioState xi = state();
  const Matrix3d R = xi.R.matrix();
  const Vector3d& x = xi.p;
  const Vector3d& v = xi.v;
  const Vector3d w = R * (gyro_meas - xi.bias_gyro);
  const Matrix3d I3 = Matrix3d::Identity();

  Linearisation lin;
  auto& F = lin.F;
  auto& B = lin.B;
  F.setZero();
  B.setZero();

  F.block<3, 3>(0, 9) = -R;
  F.block<3, 3>(3, 6) = I3;
  F.block<3, 3>(3, 9) = -Hat(x) * R;
  F.block<3, 3>(6, 0) = Hat(params_.gravity_world);
  F.block<3, 3>(6, 9) = -Hat(v) * R;
  F.block<3, 3>(6, 12) = -R;

  // delta_a = (delta upsilon, delta omega) of a = (v + x cross w, w).
  F.block<3, 3>(15, 0) = -Hat(v) + Hat(w.cross(x));
  F.block<3, 3>(15, 3) = -Hat(w);
  F.block<3, 3>(15, 6) = I3;
  F.block<3, 3>(15, 9) = -Hat(x) * R;
  F.block<3, 3>(18, 0) = -Hat(w);
  F.block<3, 3>(18, 9) = -R;
  // ad in Sophus (upsilon, omega) order: [[w^, nu^], [0, w^]].
  F.block<3, 3>(15, 15) = Hat(w);
  F.block<3, 3>(15, 18) = Hat(v + x.cross(w));
  F.block<3, 3>(18, 18) = Hat(w);

  B.block<3, 3>(0, 0) = -R;
  B.block<3, 3>(3, 0) = -Hat(x) * R;
  B.block<3, 3>(6, 0) = -Hat(v) * R;
  B.block<3, 3>(6, 3) = -R;
  B.block<3, 3>(15, 0) = -Hat(x) * R;
  B.block<3, 3>(18, 0) = -R;

  const EqvioAlgebra lambda = eqvio::Lift(xi, gyro_meas, accel_meas, params_.gravity_world);
  const Vector3d v_cam = lambda.B.head<3>();
  const Eigen::Matrix<double, 6, 6> Ad_C_inv = xi.T_world_camera().inverse().Adj();
  const Matrix3d M = PolarJacobianAtE3();

  for (std::size_t i = 0; i < ids_.size(); ++i) {
    const Vector3d& q = xi.landmarks_camera[i];
    const Matrix3d Rq = X_.Q[i].R.matrix();
    const double c = X_.Q[i].c;
    const double n2 = q.squaredNorm();
    const Matrix3d Psi = (q.dot(v_cam) / n2) * I3 + Hat(q.cross(v_cam) / n2);

    Eigen::Matrix<double, 3, 6> J;
    J << -I3, Hat(q);
    const Eigen::Matrix<double, 3, 6> W = M.transpose() * (c * Rq) * J * Ad_C_inv;

    lin.D.push_back(M.transpose() * Rq * Psi * Rq.transpose() * M);
    lin.L.push_back(W * F.middleRows<6>(15));
    lin.BL.push_back(W * B.middleRows<6>(15));
  }
  return lin;
}

Eigen::MatrixXd EqvioEstimator::StateMatrix(const Vector3d& gyro_meas,
                                            const Vector3d& accel_meas) const {
  const Linearisation lin = Linearise(gyro_meas, accel_meas);
  const int dim = static_cast<int>(Sigma_.rows());
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(dim, dim);
  A.topLeftCorner<kEqvioCoreDim, kEqvioCoreDim>() = lin.F;
  for (std::size_t i = 0; i < ids_.size(); ++i) {
    A.block<3, kEqvioCoreDim>(LandmarkOffset(i), 0) = lin.L[i];
    A.block<3, 3>(LandmarkOffset(i), LandmarkOffset(i)) = lin.D[i];
  }
  return A;
}

Eigen::MatrixXd EqvioEstimator::InputMatrix(const Vector3d& gyro_meas,
                                            const Vector3d& accel_meas) const {
  const Linearisation lin = Linearise(gyro_meas, accel_meas);
  Eigen::MatrixXd B(Sigma_.rows(), 6);
  B.topRows<kEqvioCoreDim>() = lin.B;
  for (std::size_t i = 0; i < ids_.size(); ++i) B.middleRows<3>(LandmarkOffset(i)) = lin.BL[i];
  return B;
}

void EqvioEstimator::Predict(const Vector3d& gyro_meas, const Vector3d& accel_meas, double dt) {
  const Linearisation lin = Linearise(gyro_meas, accel_meas);
  const std::size_t n = ids_.size();

  // Phi = I + A0 dt has the block structure [[F, 0], [L, D]] with D block
  // diagonal, so Phi * M costs O(dim^2 * 21) rather than O(dim^3).
  auto apply_phi = [&](const Eigen::MatrixXd& Mx) {
    Eigen::MatrixXd out = Mx;
    const auto core = Mx.topRows<kEqvioCoreDim>();
    out.topRows<kEqvioCoreDim>() += dt * (lin.F * core);
    for (std::size_t i = 0; i < n; ++i) {
      out.middleRows<3>(LandmarkOffset(i)) +=
          dt * (lin.L[i] * core + lin.D[i] * Mx.middleRows<3>(LandmarkOffset(i)));
    }
    return out;
  };
  const Eigen::MatrixXd Phi_Sigma = apply_phi(Sigma_);
  Sigma_ = apply_phi(Phi_Sigma.transpose());

  Eigen::MatrixXd B(Sigma_.rows(), 6);
  B.topRows<kEqvioCoreDim>() = lin.B;
  for (std::size_t i = 0; i < n; ++i) B.middleRows<3>(LandmarkOffset(i)) = lin.BL[i];
  Eigen::Matrix<double, 6, 6> Qc = Eigen::Matrix<double, 6, 6>::Zero();
  Qc.diagonal().head<3>().setConstant(std::pow(params_.imu.gyro_noise_density, 2));
  Qc.diagonal().tail<3>().setConstant(std::pow(params_.imu.accel_noise_density, 2));
  Sigma_ += (B * Qc * B.transpose()) * dt;

  Eigen::VectorXd m_eps = Eigen::VectorXd::Zero(Sigma_.rows());
  m_eps.segment<3>(9).setConstant(std::pow(params_.imu.gyro_random_walk, 2));
  m_eps.segment<3>(12).setConstant(std::pow(params_.imu.accel_random_walk, 2));
  m_eps.segment<6>(15).setConstant(params_.camera_process_noise);
  m_eps.tail(3 * n).setConstant(params_.landmark_process_noise);
  Sigma_.diagonal() += m_eps * dt;

  // The lift is evaluated at xi_hat before X moves, matching the Jacobians.
  X_ = X_ * eqvio::Exp(eqvio::Lift(state(), gyro_meas, accel_meas, params_.gravity_world), dt);
}

Eigen::Matrix<double, 2, 3> EqvioEstimator::OutputMatrix(const Vector3d& y_origin) {
  // Equivariant output approximation [van Goor et al. 2022, Lemma V.3]:
  // C* = 1/2 (D rho(E, y) + D rho(E, e3)) in the normal coordinates, and
  // d/dt rho(exp(t eps), y) = d/dt Exp(-t omega) y = y^ omega, so
  // C* = 1/2 (y + e3)^ [e1 e2 0]. The rotation about e3 is a stabiliser
  // direction and does not appear in eps.
  Matrix3d P = Matrix3d::Zero();
  P(0, 0) = 1;
  P(1, 1) = 1;
  const Matrix3d C3 = 0.5 * Hat(y_origin + Vector3d::UnitZ()) * P;
  return C3.topRows<2>();
}

Vector3d EqvioEstimator::BearingAtOrigin(std::size_t slot, const Vector3d& bearing) const {
  // Equivariance h(phi(X, e)) = rho(X, h(e)) with rho(Q, y) = R_Q^T y gives
  // h(e) = R_Q y: the measurement transported back to the origin.
  return X_.Q[slot].R * bearing.normalized();
}

void EqvioEstimator::AddLandmark(int id, const Vector3d& bearing, double depth) {
  // phi_V(Q, e3) = c^-1 R^T e3 = depth * bearing.
  SOT3 Q;
  Q.R = Sophus::SO3d(Eigen::Quaterniond::FromTwoVectors(bearing.normalized(), Vector3d::UnitZ()));
  Q.c = 1.0 / depth;
  X_.Q.push_back(Q);
  ids_.push_back(id);

  const int old_dim = static_cast<int>(Sigma_.rows());
  Eigen::MatrixXd grown = Eigen::MatrixXd::Zero(old_dim + 3, old_dim + 3);
  grown.topLeftCorner(old_dim, old_dim) = Sigma_;
  const double sb2 = std::pow(params_.initial_landmark_bearing_sigma, 2);
  grown.bottomRightCorner<3, 3>().diagonal()
      << sb2, sb2, std::pow(params_.initial_landmark_log_depth_sigma, 2);
  Sigma_ = std::move(grown);
}

void EqvioEstimator::RemoveLandmarks(const std::vector<bool>& keep) {
  std::vector<int> rows;
  for (int r = 0; r < kEqvioCoreDim; ++r) rows.push_back(r);
  std::vector<int> ids;
  std::vector<SOT3> Q;
  for (std::size_t i = 0; i < ids_.size(); ++i) {
    if (!keep[i]) continue;
    ids.push_back(ids_[i]);
    Q.push_back(X_.Q[i]);
    for (int k = 0; k < 3; ++k) rows.push_back(LandmarkOffset(i) + k);
  }
  const int dim = static_cast<int>(rows.size());
  Eigen::MatrixXd reduced(dim, dim);
  for (int r = 0; r < dim; ++r) {
    for (int c = 0; c < dim; ++c) reduced(r, c) = Sigma_(rows[r], rows[c]);
  }
  Sigma_ = std::move(reduced);
  ids_ = std::move(ids);
  X_.Q = std::move(Q);
}

EqvioEstimator::UpdateResult EqvioEstimator::Update(
    const std::vector<BearingMeasurement>& measurements) {
  UpdateResult result;
  std::unordered_map<int, Vector3d> bearing_of;
  for (const BearingMeasurement& m : measurements) bearing_of[m.id] = m.bearing.normalized();

  // Preprocessing (Sec. 7): drop lost landmarks, then add new ones.
  std::vector<bool> keep(ids_.size());
  for (std::size_t i = 0; i < ids_.size(); ++i) {
    keep[i] = bearing_of.count(ids_[i]) > 0;
    if (!keep[i]) ++result.num_removed;
  }
  if (result.num_removed > 0) RemoveLandmarks(keep);

  double depth = params_.initial_landmark_depth;
  if (!ids_.empty()) {
    const EqvioState xi = state();
    std::vector<double> depths;
    for (const Vector3d& q : xi.landmarks_camera) depths.push_back(q.norm());
    std::nth_element(depths.begin(), depths.begin() + depths.size() / 2, depths.end());
    depth = depths[depths.size() / 2];
  }
  std::unordered_map<int, bool> tracked;
  for (const int id : ids_) tracked[id] = true;
  for (const BearingMeasurement& m : measurements) {
    if (tracked.count(m.id)) continue;
    AddLandmark(m.id, m.bearing, depth);
    tracked[m.id] = true;
    ++result.num_added;
  }

  const double sigma2 = params_.bearing_sigma * params_.bearing_sigma;
  if (params_.outlier_chi2_threshold > 0) {
    std::vector<bool> inlier(ids_.size(), true);
    for (std::size_t i = 0; i < ids_.size(); ++i) {
      const Vector3d y0 = BearingAtOrigin(i, bearing_of[ids_[i]]);
      const Eigen::Vector2d r = (y0 - Vector3d::UnitZ()).head<2>();
      const Eigen::Matrix<double, 2, 3> C = OutputMatrix(y0);
      const int o = LandmarkOffset(i);
      const Eigen::Matrix2d S =
          C * Sigma_.block<3, 3>(o, o) * C.transpose() + sigma2 * Eigen::Matrix2d::Identity();
      if (r.dot(S.ldlt().solve(r)) > params_.outlier_chi2_threshold) {
        inlier[i] = false;
        ++result.num_outliers;
      }
    }
    if (result.num_outliers > 0) RemoveLandmarks(inlier);
  }

  const std::size_t n = ids_.size();
  result.num_used = static_cast<int>(n);
  if (n == 0) return result;

  const int dim = static_cast<int>(Sigma_.rows());
  Eigen::VectorXd r(2 * n);
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(2 * n, dim);
  for (std::size_t i = 0; i < n; ++i) {
    const Vector3d y0 = BearingAtOrigin(i, bearing_of[ids_[i]]);
    r.segment<2>(2 * i) = (y0 - Vector3d::UnitZ()).head<2>();
    C.block<2, 3>(2 * i, LandmarkOffset(i)) = OutputMatrix(y0);
  }

  const Eigen::MatrixXd SigmaCt = Sigma_ * C.transpose();
  Eigen::MatrixXd S = C * SigmaCt;
  S.diagonal().array() += sigma2;
  const Eigen::LDLT<Eigen::MatrixXd> S_ldlt(S);
  const Eigen::MatrixXd K = S_ldlt.solve(SigmaCt.transpose()).transpose();
  const Eigen::VectorXd delta = K * r;

  Sigma_ -= K * SigmaCt.transpose();
  Sigma_ = 0.5 * (Sigma_ + Sigma_.transpose()).eval();

  // eps_hat = delta means the new estimate is phi(X, vartheta^-1(delta)) =
  // phi(Z X, xi0) with Z the normal-coordinates group element of delta.
  X_ = eqvio::CorrectionFromLocalCoordinates(delta) * X_;
  return result;
}

}  // namespace vio

#include "eqvio.h"

#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "vio_simulator.h"

namespace vio {
namespace {

using Vector3d = Eigen::Vector3d;

const Vector3d kGravity(0, 0, -9.81);

Vector3d RandomVector(std::mt19937& rng, double scale) {
  std::normal_distribution<double> n(0.0, scale);
  return {n(rng), n(rng), n(rng)};
}

EqvioState RandomState(std::mt19937& rng, int num_landmarks) {
  EqvioState xi;
  xi.R = Sophus::SO3d::exp(RandomVector(rng, 1.0));
  xi.p = RandomVector(rng, 1.0);
  xi.v = RandomVector(rng, 1.0);
  xi.bias_gyro = RandomVector(rng, 0.05);
  xi.bias_accel = RandomVector(rng, 0.1);
  xi.T_body_camera = Sophus::SE3d(Sophus::SO3d::exp(RandomVector(rng, 0.5)), RandomVector(rng, 0.2));
  for (int i = 0; i < num_landmarks; ++i) {
    xi.landmark_ids.push_back(10 + i);
    xi.landmarks_camera.push_back(Vector3d(0, 0, 2.0) + RandomVector(rng, 0.7));
  }
  return xi;
}

EqvioGroup RandomGroup(std::mt19937& rng, int num_landmarks) {
  EqvioGroup X;
  Vector9d a;
  a << RandomVector(rng, 1.0), RandomVector(rng, 1.0), RandomVector(rng, 1.0);
  X.A = SE23::exp(a);
  X.beta << RandomVector(rng, 0.1), RandomVector(rng, 0.1);
  Vector6d b;
  b << RandomVector(rng, 0.5), RandomVector(rng, 0.5);
  X.B = Sophus::SE3d::exp(b);
  for (int i = 0; i < num_landmarks; ++i) {
    X.Q.push_back(SOT3::exp(RandomVector(rng, 1.0), RandomVector(rng, 0.3).x()));
  }
  return X;
}

void ExpectStatesNear(const EqvioState& a, const EqvioState& b, double tol) {
  EXPECT_LT((a.R.inverse() * b.R).log().norm(), tol);
  EXPECT_LT((a.p - b.p).norm(), tol);
  EXPECT_LT((a.v - b.v).norm(), tol);
  EXPECT_LT((a.bias_gyro - b.bias_gyro).norm(), tol);
  EXPECT_LT((a.bias_accel - b.bias_accel).norm(), tol);
  EXPECT_LT((a.T_body_camera.inverse() * b.T_body_camera).log().norm(), tol);
  ASSERT_EQ(a.landmarks_camera.size(), b.landmarks_camera.size());
  for (std::size_t i = 0; i < a.landmarks_camera.size(); ++i) {
    EXPECT_LT((a.landmarks_camera[i] - b.landmarks_camera[i]).norm(), tol);
  }
}

// One explicit-Euler step of the VI-SLAM dynamics, eq. (5), written out
// independently of the lift: inputs are the true (noise-free) IMU readings.
EqvioState FlowTruth(const EqvioState& xi, const Vector3d& gyro, const Vector3d& accel, double tau) {
  EqvioState out = xi;
  const Vector3d omega = gyro - xi.bias_gyro;
  const Vector3d a = accel - xi.bias_accel;
  out.R = xi.R * Sophus::SO3d::exp(omega * tau);
  out.p = xi.p + xi.v * tau;
  out.v = xi.v + (xi.R * a + kGravity) * tau;
  Vector6d U;
  U << xi.R.inverse() * xi.v, omega;
  const Vector6d cam = xi.T_body_camera.inverse().Adj() * U;
  for (Vector3d& q : out.landmarks_camera) {
    q += tau * (-cam.tail<3>().cross(q) - cam.head<3>());
  }
  return out;
}

TEST(EqvioGroups, SE23ExpLogRoundTrip) {
  std::mt19937 rng(1);
  for (int trial = 0; trial < 20; ++trial) {
    Vector9d xi;
    xi << RandomVector(rng, 1.0), RandomVector(rng, 2.0), RandomVector(rng, 2.0);
    EXPECT_LT((SE23::log(SE23::exp(xi)) - xi).norm(), 1e-9);
  }
}

TEST(EqvioGroups, SE23PoseMatchesSophusSE3) {
  // SE_2(3)'s (R, x) block is SE(3), so exp must agree with Sophus there.
  std::mt19937 rng(2);
  Vector9d xi;
  xi << RandomVector(rng, 1.0), RandomVector(rng, 1.0), RandomVector(rng, 1.0);
  Vector6d se3;
  se3 << xi.segment<3>(3), xi.segment<3>(0);
  const Sophus::SE3d expected = Sophus::SE3d::exp(se3);
  EXPECT_LT((SE23::exp(xi).pose().inverse() * expected).log().norm(), 1e-12);
}

TEST(EqvioGroups, SOT3ActionIsARightAction) {
  std::mt19937 rng(3);
  const SOT3 Q1 = SOT3::exp(RandomVector(rng, 1.0), 0.3);
  const SOT3 Q2 = SOT3::exp(RandomVector(rng, 1.0), -0.7);
  const Vector3d q = RandomVector(rng, 1.0);
  EXPECT_LT((Q2.Act(Q1.Act(q)) - (Q1 * Q2).Act(q)).norm(), 1e-12);
}

TEST(EqvioGeometry, PolarCoordinatesRoundTrip) {
  std::mt19937 rng(4);
  for (int trial = 0; trial < 50; ++trial) {
    const Vector3d q = Vector3d(0, 0, 1.5) + RandomVector(rng, 1.0);
    EXPECT_LT((eqvio::FromPolarCoordinates(eqvio::PolarCoordinates(q)) - q).norm(), 1e-12);
  }
  EXPECT_LT(eqvio::PolarCoordinates(Vector3d(0, 0, 1)).norm(), 1e-15);
}

TEST(EqvioGeometry, ActionIsARightAction) {
  std::mt19937 rng(5);
  const EqvioState xi = RandomState(rng, 4);
  const EqvioGroup X1 = RandomGroup(rng, 4);
  const EqvioGroup X2 = RandomGroup(rng, 4);
  ExpectStatesNear(eqvio::Act(X2, eqvio::Act(X1, xi)), eqvio::Act(X1 * X2, xi), 1e-9);
  ExpectStatesNear(eqvio::Act(X1.inverse(), eqvio::Act(X1, xi)), xi, 1e-9);
}

TEST(EqvioGeometry, MeasurementIsEquivariant) {
  // Lemma 5.5: h(phi(X, xi)) = rho(X, h(xi)) with rho(Q, y) = R_Q^T y.
  std::mt19937 rng(6);
  const EqvioState xi = RandomState(rng, 5);
  const EqvioGroup X = RandomGroup(rng, 5);
  const EqvioState moved = eqvio::Act(X, xi);
  for (std::size_t i = 0; i < xi.landmarks_camera.size(); ++i) {
    const Vector3d expected = X.Q[i].R.inverse() * xi.landmarks_camera[i].normalized();
    EXPECT_LT((moved.landmarks_camera[i].normalized() - expected).norm(), 1e-12);
  }
}

TEST(EqvioGeometry, ActionPreservesWorldLandmarksUnderPureIMUGroupElements) {
  // With B = I and Q_i = I, the camera pose becomes (P P_A)(P_A^-1 T) = P T:
  // A moves the body but neither the camera nor the world landmarks.
  std::mt19937 rng(7);
  const EqvioState xi = RandomState(rng, 3);
  EqvioGroup X = RandomGroup(rng, 3);
  X.B = Sophus::SE3d();
  for (SOT3& Q : X.Q) Q = SOT3();
  const EqvioState moved = eqvio::Act(X, xi);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_LT((moved.LandmarkWorld(i) - xi.LandmarkWorld(i)).norm(), 1e-9);
  }
}

TEST(EqvioGeometry, LocalCoordinatesRoundTrip) {
  std::mt19937 rng(8);
  const EqvioState e = RandomState(rng, 4);
  const Eigen::VectorXd eps = eqvio::LocalCoordinates(e);
  ExpectStatesNear(eqvio::FromLocalCoordinates(eps, e.landmark_ids), e, 1e-9);
}

TEST(EqvioGeometry, CorrectionIsNormalCoordinates) {
  // vartheta(phi(Z(eps), xi0)) == eps: the update's group element lands
  // exactly on the requested local-coordinate correction.
  std::mt19937 rng(9);
  Eigen::VectorXd eps(kEqvioCoreDim + 9);
  for (int i = 0; i < eps.size(); ++i) eps(i) = RandomVector(rng, 0.3).x();
  const std::vector<int> ids = {1, 2, 3};
  const EqvioState moved =
      eqvio::Act(eqvio::CorrectionFromLocalCoordinates(eps), eqvio::Origin(ids));
  EXPECT_LT((eqvio::LocalCoordinates(moved) - eps).norm(), 1e-9);
}

TEST(EqvioGeometry, LiftSatisfiesLiftCondition) {
  // eq. (26): D phi_xi(E) Lambda(xi, u) = f_u(xi).
  std::mt19937 rng(10);
  const EqvioState xi = RandomState(rng, 4);
  const Vector3d gyro = RandomVector(rng, 0.5);
  const Vector3d accel = RandomVector(rng, 3.0);
  const EqvioAlgebra lambda = eqvio::Lift(xi, gyro, accel, kGravity);
  const double tau = 1e-6;
  const EqvioState plus = eqvio::Act(eqvio::Exp(lambda, tau), xi);
  const EqvioState minus = eqvio::Act(eqvio::Exp(lambda, -tau), xi);
  const EqvioState f_plus = FlowTruth(xi, gyro, accel, tau);
  const EqvioState f_minus = FlowTruth(xi, gyro, accel, -tau);

  auto rate = [&](const Vector3d& p, const Vector3d& m) { return Vector3d((p - m) / (2 * tau)); };
  EXPECT_LT((rate(plus.p, minus.p) - rate(f_plus.p, f_minus.p)).norm(), 1e-6);
  EXPECT_LT((rate(plus.v, minus.v) - rate(f_plus.v, f_minus.v)).norm(), 1e-6);
  EXPECT_LT(((minus.R.inverse() * plus.R).log() - (f_minus.R.inverse() * f_plus.R).log()).norm(),
            1e-9);
  EXPECT_LT((plus.T_body_camera.inverse() * xi.T_body_camera).log().norm(), 1e-9);
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_LT((rate(plus.landmarks_camera[i], minus.landmarks_camera[i]) -
               rate(f_plus.landmarks_camera[i], f_minus.landmarks_camera[i]))
                  .norm(),
              1e-6);
  }
}

// An estimator with landmarks and a generic observer state: landmarks are
// added by an update, then a few predictions rotate their SOT(3) gauge.
EqvioEstimator MakeGenericEstimator(std::mt19937& rng) {
  EqvioState init = RandomState(rng, 0);
  EqvioParams params;
  params.imu.gyro_noise_density = 1e-3;
  params.imu.accel_noise_density = 1e-2;
  EqvioEstimator est(init, Eigen::MatrixXd::Identity(kEqvioCoreDim, kEqvioCoreDim) * 1e-2, params);
  std::vector<BearingMeasurement> meas;
  for (int i = 0; i < 4; ++i) {
    meas.push_back({i, (Vector3d(0, 0, 1) + RandomVector(rng, 0.3)).normalized()});
  }
  est.Update(meas);
  for (int k = 0; k < 5; ++k) est.Predict(RandomVector(rng, 0.5), RandomVector(rng, 3.0), 0.05);
  return est;
}

// Finite-difference eps_dot of the true error dynamics: the true state
// follows eq. (5) with inputs (u - n), the observer follows X_dot = X Lambda.
Eigen::VectorXd NumericalErrorRate(const EqvioEstimator& est, const Eigen::VectorXd& eps,
                                   const Vector3d& gyro, const Vector3d& accel,
                                   const Eigen::Matrix<double, 6, 1>& noise) {
  const EqvioGroup& X = est.observer();
  const std::vector<int>& ids = est.landmark_ids();
  const EqvioState xi = eqvio::Act(X, eqvio::FromLocalCoordinates(eps, ids));
  const EqvioState xi_hat = eqvio::Act(X, eqvio::Origin(ids));
  const EqvioAlgebra lambda_hat = eqvio::Lift(xi_hat, gyro, accel, kGravity);
  const double tau = 1e-5;
  auto eps_at = [&](double t) {
    const EqvioState xi_t = FlowTruth(xi, gyro - noise.head<3>(), accel - noise.tail<3>(), t);
    const EqvioGroup X_t = X * eqvio::Exp(lambda_hat, t);
    return eqvio::LocalCoordinates(eqvio::Act(X_t.inverse(), xi_t));
  };
  return (eps_at(tau) - eps_at(-tau)) / (2 * tau);
}

TEST(EqvioFilter, StateMatrixMatchesNumerical) {
  std::mt19937 rng(11);
  const EqvioEstimator est = MakeGenericEstimator(rng);
  const Vector3d gyro = RandomVector(rng, 0.5);
  const Vector3d accel = RandomVector(rng, 3.0);
  const Eigen::MatrixXd A = est.StateMatrix(gyro, accel);
  const int dim = static_cast<int>(A.rows());
  ASSERT_EQ(dim, kEqvioCoreDim + 12);

  const double h = 1e-4;
  const Eigen::Matrix<double, 6, 1> no_noise = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::MatrixXd A_num(dim, dim);
  for (int j = 0; j < dim; ++j) {
    Eigen::VectorXd e = Eigen::VectorXd::Zero(dim);
    e(j) = h;
    A_num.col(j) = (NumericalErrorRate(est, e, gyro, accel, no_noise) -
                    NumericalErrorRate(est, -e, gyro, accel, no_noise)) /
                   (2 * h);
  }
  EXPECT_LT((A - A_num).cwiseAbs().maxCoeff(), 1e-4 * (1.0 + A_num.cwiseAbs().maxCoeff()))
      << "analytic:\n" << A << "\nnumerical:\n" << A_num;
}

TEST(EqvioFilter, InputMatrixMatchesNumerical) {
  std::mt19937 rng(12);
  const EqvioEstimator est = MakeGenericEstimator(rng);
  const Vector3d gyro = RandomVector(rng, 0.5);
  const Vector3d accel = RandomVector(rng, 3.0);
  const Eigen::MatrixXd B = est.InputMatrix(gyro, accel);
  const int dim = static_cast<int>(B.rows());
  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(dim);

  const double h = 1e-4;
  Eigen::MatrixXd B_num(dim, 6);
  for (int j = 0; j < 6; ++j) {
    Eigen::Matrix<double, 6, 1> n = Eigen::Matrix<double, 6, 1>::Zero();
    n(j) = h;
    B_num.col(j) = (NumericalErrorRate(est, zero, gyro, accel, n) -
                    NumericalErrorRate(est, zero, gyro, accel, -n)) /
                   (2 * h);
  }
  EXPECT_LT((B - B_num).cwiseAbs().maxCoeff(), 1e-4 * (1.0 + B_num.cwiseAbs().maxCoeff()))
      << "analytic:\n" << B << "\nnumerical:\n" << B_num;
}

TEST(EqvioFilter, EquivariantOutputApproximationIsThirdOrder) {
  // y - h(xi_hat) = C* eps + O(|eps|^3) (Sec. 6), versus O(|eps|^2) for the
  // ordinary Jacobian C0 = C*(e3) = e3^ [e1 e2 0]. In R^3, halving eps should
  // cut the C* residual by ~8 and the C0 residual by ~4.
  const Vector3d dir = Vector3d(0.6, -0.8, 0.0).normalized() + Vector3d(0, 0, 0.5);
  Eigen::Matrix3d P = Eigen::Matrix3d::Zero();
  P(0, 0) = P(1, 1) = 1;
  auto ambient_errors = [&](double t) {
    const Vector3d z = t * dir;
    const Vector3d y = eqvio::FromPolarCoordinates(z).normalized();
    const Vector3d r = y - Vector3d::UnitZ();
    const Vector3d c_star = 0.5 * Sophus::SO3d::hat(y + Vector3d::UnitZ()) * P * z;
    const Vector3d c_zero = Sophus::SO3d::hat(Vector3d::UnitZ()) * P * z;
    return std::make_pair((r - c_star).norm(), (r - c_zero).norm());
  };
  const auto [star_big, plain_big] = ambient_errors(0.2);
  const auto [star_small, plain_small] = ambient_errors(0.1);
  EXPECT_NEAR(star_big / star_small, 8.0, 0.5);
  EXPECT_NEAR(plain_big / plain_small, 4.0, 0.5);

  // The filter uses the e1/e2 projection, which already drops C0's
  // second-order term (-|omega|^2 e3 / 2 lies along e3). Both are then third
  // order, but C*'s constant is half of C0's: |omega|^3/12 vs |omega|^3/6.
  const Vector3d z = 0.1 * dir;
  const Vector3d y = eqvio::FromPolarCoordinates(z).normalized();
  const Eigen::Vector2d r = (y - Vector3d::UnitZ()).head<2>();
  const double star = (r - EqvioEstimator::OutputMatrix(y) * z).norm();
  const double plain = (r - EqvioEstimator::OutputMatrix(Vector3d::UnitZ()) * z).norm();
  EXPECT_NEAR(star / plain, 0.5, 0.05);
}

TEST(EqvioFilter, UpdateManagesLandmarks) {
  EqvioParams params;
  EqvioEstimator est(EqvioState(), Eigen::MatrixXd::Identity(kEqvioCoreDim, kEqvioCoreDim) * 1e-4,
                     params);
  auto r = est.Update({{1, Vector3d(0.1, 0, 1).normalized()}, {2, Vector3d(0, 0.1, 1).normalized()}});
  EXPECT_EQ(r.num_added, 2);
  EXPECT_EQ(est.covariance().rows(), kEqvioCoreDim + 6);
  // The new landmarks sit on their measured bearings at the default depth.
  const EqvioState s = est.state();
  EXPECT_NEAR(s.landmarks_camera[0].norm(), params.initial_landmark_depth, 1e-12);
  EXPECT_LT((s.landmarks_camera[0].normalized() - Vector3d(0.1, 0, 1).normalized()).norm(), 1e-12);

  r = est.Update({{2, Vector3d(0, 0.1, 1).normalized()}, {3, Vector3d(0, 0, 1)}});
  EXPECT_EQ(r.num_removed, 1);
  EXPECT_EQ(r.num_added, 1);
  EXPECT_EQ(est.landmark_ids(), (std::vector<int>{2, 3}));
  EXPECT_EQ(est.covariance().rows(), kEqvioCoreDim + 6);
}

TEST(EqvioFilter, ExactMeasurementsLeaveTheTrueStateFixed) {
  std::mt19937 rng(13);
  EqvioState truth = RandomState(rng, 5);
  EqvioParams params;
  EqvioEstimator est(truth, Eigen::MatrixXd::Identity(kEqvioCoreDim, kEqvioCoreDim) * 1e-2, params);
  std::vector<BearingMeasurement> meas;
  for (std::size_t i = 0; i < 5; ++i) {
    meas.push_back({truth.landmark_ids[i], truth.landmarks_camera[i].normalized()});
  }
  est.Update(meas);
  const Eigen::VectorXd before = est.ErrorCoordinates(truth);
  est.Update(meas);
  const Eigen::VectorXd after = est.ErrorCoordinates(truth);
  // Zero residual, zero correction: the (wrong-depth) landmarks don't move
  // the rest of the state either.
  EXPECT_LT((after - before).norm(), 1e-9);
}

TEST(EqvioFilter, TracksSimulatedTrajectory) {
  VioSimConfig config;
  config.duration_s = 20.0;
  config.noise.gyro_noise_density = 1.7e-4;
  config.noise.accel_noise_density = 2.0e-3;
  config.noise.gyro_random_walk = 1.9e-5;
  config.noise.accel_random_walk = 3.0e-3;
  config.initial_bias_gyro = Vector3d(0.01, -0.02, 0.015);
  config.initial_bias_accel = Vector3d(0.05, -0.03, 0.04);
  config.max_tracked_features = 20;
  const VioSimulation sim = SimulateVio(config);

  EqvioState init = TruthState(sim, 0);
  init.bias_gyro.setZero();
  init.bias_accel.setZero();
  init.landmark_ids.clear();
  init.landmarks_camera.clear();
  Eigen::Matrix<double, 9, 9> nav = Eigen::Matrix<double, 9, 9>::Zero();
  nav.diagonal() << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
  Eigen::Matrix<double, 6, 6> bias = Eigen::Matrix<double, 6, 6>::Zero();
  bias.diagonal() << 1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2;
  const Eigen::Matrix<double, 6, 6> extrinsic = Eigen::Matrix<double, 6, 6>::Identity() * 1e-8;

  EqvioParams params;
  params.imu = config.noise;
  params.bearing_sigma = config.pixel_sigma / config.focal_px;
  EqvioEstimator est(init, eqvio::InitialCovariance(init, nav, bias, extrinsic), params);
  EskfEstimator dead_reckoning(EskfState{init.p, init.v, init.R}, Eigen::Matrix<double, 15, 15>::Zero(),
                               config.noise, config.gravity_world);

  const double dt = 1.0 / config.imu_hz;
  std::size_t next_frame = 0;
  double sq_err = 0;
  int samples = 0;
  for (std::size_t k = 0; k < sim.imu.size(); ++k) {
    if (next_frame < sim.frames.size() && sim.frames[next_frame].imu_index == k) {
      est.Update(sim.frames[next_frame].bearings);
      ++next_frame;
      const EqvioState s = est.state();
      sq_err += (s.p - sim.imu[k].p_true).squaredNorm();
      ++samples;
    }
    if (k + 1 < sim.imu.size()) {
      est.Predict(sim.imu[k].gyro_meas, sim.imu[k].accel_meas, dt);
      dead_reckoning.Predict(sim.imu[k].gyro_meas, sim.imu[k].accel_meas, dt);
    }
  }
  const ImuTrajectorySample& last = sim.imu.back();
  const EqvioState s = est.state();
  const double rmse = std::sqrt(sq_err / samples);
  const double att_err = (s.R.inverse() * last.R_true).log().norm();
  const double dr_err = (dead_reckoning.state().p - last.p_true).norm();
  std::cout << "position RMSE " << rmse << " m, final attitude error " << att_err
            << " rad, gyro bias error " << (s.bias_gyro - last.gyro_bias_true).norm()
            << ", accel bias error " << (s.bias_accel - last.accel_bias_true).norm()
            << ", dead-reckoning final error " << dr_err << " m\n";
  EXPECT_LT(rmse, 0.1);
  EXPECT_LT(att_err, 0.02);
  EXPECT_LT((s.bias_gyro - last.gyro_bias_true).norm(), 0.005);
  EXPECT_LT((s.bias_accel - last.accel_bias_true).norm(), 0.05);
  EXPECT_GT(dr_err, 10 * rmse);
}

}  // namespace
}  // namespace vio

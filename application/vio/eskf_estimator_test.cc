#include "eskf_estimator.h"

#include <cmath>

#include "gtest/gtest.h"

namespace vio {
namespace {

Eigen::Matrix<double, 15, 15> ZeroCov() {
  return Eigen::Matrix<double, 15, 15>::Zero();
}

Eigen::Matrix<double, 15, 15> SmallCov() {
  return Eigen::Matrix<double, 15, 15>::Identity() * 1e-6;
}

ImuNoiseParams SampleNoise() {
  // The real EuRoC imu0 densities, so the covariance-scale tests exercise
  // realistic magnitudes.
  ImuNoiseParams n;
  n.gyro_noise_density = 1.6968e-04;
  n.gyro_random_walk = 1.9393e-05;
  n.accel_noise_density = 2.0000e-3;
  n.accel_random_walk = 3.0000e-3;
  return n;
}

}  // namespace

TEST(EskfEstimator, PredictStationaryHoldsPositionAndVelocity) {
  EskfState x0;
  EskfEstimator est(x0, SmallCov(), SampleNoise());

  for (int i = 0; i < 1000; ++i) {
    est.Predict(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 9.81), 0.005);
  }

  EXPECT_LT(est.state().p.norm(), 1e-9);
  EXPECT_LT(est.state().v.norm(), 1e-9);
}

TEST(EskfEstimator, PredictConstantForwardAcceleration) {
  EskfState x0;
  EskfEstimator est(x0, SmallCov(), SampleNoise());

  const double dt = 0.001;
  const int steps = 1000;  // T = 1s
  for (int i = 0; i < steps; ++i) {
    est.Predict(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 9.81), dt);
  }

  // gyro=0 keeps R fixed at identity, so a_world=(1,0,0) is constant every
  // step: this discretization is exact Euler integration of a constant
  // acceleration, so v is exact; p accumulates the usual O(dt) Euler error.
  EXPECT_NEAR(est.state().v.x(), 1.0, 1e-6);
  EXPECT_NEAR(est.state().p.x(), 0.5, 1e-3);
}

TEST(EskfEstimator, PredictConstantAngularRateMatchesExpMap) {
  EskfState x0;
  EskfEstimator est(x0, SmallCov(), SampleNoise());

  const double dt = 0.001;
  const int steps = 2000;  // T = 2s
  for (int i = 0; i < steps; ++i) {
    // accel=(0,0,9.81) keeps p,v near zero throughout so this test isolates
    // rotation only; the exact value doesn't matter for what's asserted.
    est.Predict(Eigen::Vector3d(0, 0, 0.5), Eigen::Vector3d(0, 0, 9.81), dt);
  }

  // Repeated composition of Exp(omega*dt) about a FIXED axis has no
  // discretization error: R after N steps is exactly Exp(omega*N*dt).
  const Sophus::SO3d expected = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, 0.5 * steps * dt));
  EXPECT_LT((est.state().R.inverse() * expected).log().norm(), 1e-9);
}

TEST(EskfEstimator, PredictSingleStepCovarianceMatchesClosedForm) {
  EskfState x0;
  const ImuNoiseParams noise = SampleNoise();
  EskfEstimator est(x0, ZeroCov(), noise);

  // omega=0, a=0 exactly (bias-corrected measurement equals zero-bias
  // state), dt=1.0 chosen so the sigma^2*dt scaling is easy to check
  // numerically without extra arithmetic.
  est.Predict(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0), 1.0);

  const Eigen::Matrix<double, 15, 15>& P = est.covariance();
  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();

  EXPECT_LT((P.block<3, 3>(0, 0)).norm(), 1e-12);
  EXPECT_LT((P.block<3, 3>(3, 3) - noise.accel_noise_density * noise.accel_noise_density * I3)
               .norm(),
           1e-15);
  EXPECT_LT((P.block<3, 3>(6, 6) - noise.gyro_noise_density * noise.gyro_noise_density * I3)
               .norm(),
           1e-15);
  EXPECT_LT((P.block<3, 3>(9, 9) - noise.gyro_random_walk * noise.gyro_random_walk * I3).norm(),
           1e-18);
  EXPECT_LT(
      (P.block<3, 3>(12, 12) - noise.accel_random_walk * noise.accel_random_walk * I3).norm(),
      1e-15);
}

TEST(EskfEstimator, PredictKeepsCovarianceSymmetricAndPsd) {
  EskfState x0;
  EskfEstimator est(x0, SmallCov(), SampleNoise());

  for (int i = 0; i < 200; ++i) {
    const double t = 0.01 * i;
    est.Predict(Eigen::Vector3d(0.3 * std::sin(t), 0.1, -0.2 * std::cos(t)),
               Eigen::Vector3d(0.5, -0.3, 9.81 + 0.2 * std::sin(2 * t)), 0.005);

    const Eigen::Matrix<double, 15, 15>& P = est.covariance();
    EXPECT_LT((P - P.transpose()).norm(), 1e-9);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15>> solver(P);
    EXPECT_GE(solver.eigenvalues().minCoeff(), -1e-9);
  }
}

TEST(EskfEstimator, UpdatePerfectMeasurementLeavesStateEssentiallyUnchanged) {
  EskfState x0;
  EskfEstimator est(x0, SmallCov(), SampleNoise());

  est.SetRelativePoseAnchor();
  const Sophus::SO3d anchor_R = est.state().R;
  const Eigen::Vector3d anchor_p = est.state().p;

  for (int i = 0; i < 50; ++i) {
    est.Predict(Eigen::Vector3d(0.05, -0.02, 0.03), Eigen::Vector3d(0.4, 0.1, 9.81), 0.005);
  }

  const Sophus::SO3d curr_R = est.state().R;
  const Eigen::Vector3d curr_p = est.state().p;
  const Sophus::SO3d R_h = anchor_R.inverse() * curr_R;
  const Eigen::Vector3d t_h = anchor_R.inverse() * (curr_p - anchor_p);
  RelativePoseMeasurement z;
  z.T_prevbody_currbody = Sophus::SE3d(R_h, t_h);

  const double trace_before = est.covariance().trace();
  const EskfState before = est.state();
  const bool applied = est.UpdateRelativePose(z);
  ASSERT_TRUE(applied);

  EXPECT_LT((est.state().p - before.p).norm(), 1e-9);
  EXPECT_LT((est.state().v - before.v).norm(), 1e-9);
  EXPECT_LT((est.state().R.inverse() * before.R).log().norm(), 1e-9);
  EXPECT_LT((est.state().bias_gyro - before.bias_gyro).norm(), 1e-9);
  EXPECT_LT((est.state().bias_accel - before.bias_accel).norm(), 1e-9);
  EXPECT_LT(est.covariance().trace(), trace_before);
}

TEST(EskfEstimator, UpdateInjectedErrorCorrectsTowardTruth) {
  EskfState x0;
  EskfEstimator est(x0, SmallCov(), SampleNoise());

  est.SetRelativePoseAnchor();

  // Stationary true trajectory: gyro=0, accel cancels gravity exactly, so
  // the ideal (unperturbed) relative pose from anchor to now is identity.
  const Eigen::Vector3d stationary_accel(0, 0, 9.81);
  for (int i = 0; i < 19; ++i) {
    est.Predict(Eigen::Vector3d(0, 0, 0), stationary_accel, 0.01);
  }
  // Inject a real disturbance on the final step only: a brief unmodeled
  // acceleration bump the true trajectory (and hence the measurement z
  // below) never saw, so the filter's current state genuinely diverges
  // from truth -- not a synthetic private-state hack.
  est.Predict(Eigen::Vector3d(0, 0, 0), stationary_accel + Eigen::Vector3d(4.0, 0, 0), 0.01);

  const Eigen::Vector3d p_before = est.state().p;
  EXPECT_GT(p_before.norm(), 1e-4);  // sanity: the bump actually moved p

  // The true relative pose (anchor to now, had the bump not happened) is
  // identity. Use a tight measurement noise here -- not the default
  // 0.02m/0.01rad -- so the update actually trusts this single measurement
  // strongly; with only 20 Predict steps of the (tiny) real EuRoC noise
  // densities behind it, the filter's own covariance is far smaller than a
  // 0.02m assumption, so a default-noise measurement is correctly barely
  // trusted (that's the right Bayesian behavior, not a bug) and wouldn't
  // exercise "corrects toward truth" the way this test needs to.
  RelativePoseMeasurement z;
  z.T_prevbody_currbody = Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d::Zero());
  z.sigma_translation_m = 1e-8;
  z.sigma_rotation_rad = 1e-8;

  ASSERT_TRUE(est.UpdateRelativePose(z));
  const Eigen::Vector3d p_after = est.state().p;

  // p_true (relative to the anchor, in world frame the anchor is the
  // origin) is 0, so |p_after| is directly the post-update error.
  EXPECT_LT(p_after.norm(), 0.5 * p_before.norm());
}

TEST(EskfEstimator, UpdateWithoutAnchorIsNoOp) {
  EskfState x0;
  EskfEstimator est(x0, SmallCov(), SampleNoise());

  const EskfState before = est.state();
  const Eigen::Matrix<double, 15, 15> P_before = est.covariance();

  RelativePoseMeasurement z;  // default: identity relative pose
  EXPECT_FALSE(est.UpdateRelativePose(z));

  EXPECT_EQ(est.state().p, before.p);
  EXPECT_EQ(est.state().v, before.v);
  EXPECT_EQ(est.covariance(), P_before);
}

}  // namespace vio

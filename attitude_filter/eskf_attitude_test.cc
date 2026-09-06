#include "eskf_attitude.h"

#include <cmath>

#include "gtest/gtest.h"

namespace attitude_filter {
namespace {

Eigen::Matrix<double, 6, 6> ZeroCov() { return Eigen::Matrix<double, 6, 6>::Zero(); }

Eigen::Matrix<double, 6, 6> SmallCov() {
  return Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
}

AttitudeImuNoiseParams SampleNoise() {
  AttitudeImuNoiseParams n;
  n.gyro_noise_density = 0.05;   // rad/s / sqrt(Hz), equivalent to ~0.005 rad/s at 100 Hz
  n.gyro_random_walk = 0.0002;   // rad/s^2 / sqrt(Hz)
  return n;
}

}  // namespace

TEST(AttitudeEskf, PredictStationaryHoldsOrientation) {
  AttitudeEskfState x0;
  AttitudeEskf est(x0, SmallCov(), SampleNoise());

  for (int i = 0; i < 1000; ++i) {
    est.Predict(Eigen::Vector3d(0, 0, 0), 0.005);
  }

  EXPECT_LT(est.state().R.log().norm(), 1e-9);
}

TEST(AttitudeEskf, PredictConstantAngularRateMatchesExpMap) {
  AttitudeEskfState x0;
  AttitudeEskf est(x0, SmallCov(), SampleNoise());

  const double dt = 0.001;
  const int steps = 2000;  // T = 2s
  for (int i = 0; i < steps; ++i) {
    est.Predict(Eigen::Vector3d(0, 0, 0.5), dt);
  }

  // Repeated composition of Exp(omega*dt) about a FIXED axis has no
  // discretization error: R after N steps is exactly Exp(omega*N*dt).
  const Sophus::SO3d expected = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, 0.5 * steps * dt));
  EXPECT_LT((est.state().R.inverse() * expected).log().norm(), 1e-9);
}

TEST(AttitudeEskf, PredictSingleStepCovarianceMatchesClosedForm) {
  AttitudeEskfState x0;
  const AttitudeImuNoiseParams noise = SampleNoise();
  AttitudeEskf est(x0, ZeroCov(), noise);

  // omega=0 exactly (bias-corrected measurement equals zero-bias state),
  // dt=1.0 chosen so the sigma^2*dt scaling is easy to check numerically.
  est.Predict(Eigen::Vector3d(0, 0, 0), 1.0);

  const Eigen::Matrix<double, 6, 6>& P = est.covariance();
  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();

  EXPECT_LT((P.block<3, 3>(0, 0) - noise.gyro_noise_density * noise.gyro_noise_density * I3)
               .norm(),
           1e-15);
  EXPECT_LT((P.block<3, 3>(3, 3) - noise.gyro_random_walk * noise.gyro_random_walk * I3).norm(),
           1e-18);
}

TEST(AttitudeEskf, PredictKeepsCovarianceSymmetricAndPsd) {
  AttitudeEskfState x0;
  AttitudeEskf est(x0, SmallCov(), SampleNoise());

  for (int i = 0; i < 200; ++i) {
    const double t = 0.01 * i;
    est.Predict(Eigen::Vector3d(0.3 * std::sin(t), 0.1, -0.2 * std::cos(t)), 0.005);

    const Eigen::Matrix<double, 6, 6>& P = est.covariance();
    EXPECT_LT((P - P.transpose()).norm(), 1e-9);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(P);
    EXPECT_GE(solver.eigenvalues().minCoeff(), -1e-9);
  }
}

TEST(AttitudeEskf, UpdatePerfectMeasurementLeavesStateEssentiallyUnchanged) {
  AttitudeEskfState x0;
  AttitudeEskf est(x0, SmallCov(), SampleNoise());

  for (int i = 0; i < 50; ++i) {
    est.Predict(Eigen::Vector3d(0.05, -0.02, 0.03), 0.005);
  }

  const AttitudeEskfState before = est.state();
  VectorMeasurement z;
  z.world_reference = Eigen::Vector3d(0, 0, -1);
  z.body_measured = before.R.inverse() * z.world_reference;

  const double trace_before = est.covariance().trace();
  est.UpdateVector(z);

  EXPECT_LT((est.state().R.inverse() * before.R).log().norm(), 1e-9);
  EXPECT_LT((est.state().bias_gyro - before.bias_gyro).norm(), 1e-9);
  EXPECT_LT(est.covariance().trace(), trace_before);
}

TEST(AttitudeEskf, UpdateInjectedErrorCorrectsTowardTruth) {
  // Filter starts with a real orientation error (not the true attitude);
  // the truth is identity.
  AttitudeEskfState x0;
  x0.R = Sophus::SO3d::exp(Eigen::Vector3d(0.2, -0.1, 0.15));
  AttitudeEskf est(x0, SmallCov(), SampleNoise());

  const double error_before = est.state().R.log().norm();
  ASSERT_GT(error_before, 1e-3);

  VectorMeasurement gravity;
  gravity.world_reference = Eigen::Vector3d(0, 0, -1);
  gravity.body_measured = gravity.world_reference;  // true R is identity
  gravity.sigma = 1e-6;
  est.UpdateVector(gravity);

  VectorMeasurement mag;
  mag.world_reference = Eigen::Vector3d(0.45, 0.05, 0.89).normalized();
  mag.body_measured = mag.world_reference;  // true R is identity
  mag.sigma = 1e-6;
  est.UpdateVector(mag);

  EXPECT_LT(est.state().R.log().norm(), 0.1 * error_before);
}

TEST(AttitudeEskf, GravityAloneDoesNotObserveHeadingAboutItsOwnAxis) {
  // A heading (yaw about the gravity axis) error is invisible to a gravity
  // measurement alone: gravity points along the rotation axis of the error,
  // so R^{-1}*gravity_world is unchanged by that error and the residual is
  // ~zero regardless of how large the heading error is.
  AttitudeEskfState x0;
  x0.R = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, 0.7));  // yaw about z (=gravity axis) only
  AttitudeEskf est(x0, SmallCov(), SampleNoise());

  VectorMeasurement gravity;
  gravity.world_reference = Eigen::Vector3d(0, 0, -1);
  gravity.body_measured = Eigen::Vector3d(0, 0, -1);  // true R is identity
  gravity.sigma = 1e-6;
  est.UpdateVector(gravity);

  EXPECT_GT(est.state().R.log().norm(), 0.5);
}

}  // namespace attitude_filter

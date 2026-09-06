#include "eqf_attitude.h"

#include "gtest/gtest.h"

namespace eqf_attitude {
namespace {

Eigen::Matrix<double, 3, 3> SmallCov() {
  return Eigen::Matrix<double, 3, 3>::Identity() * 1e-6;
}

EqfImuNoiseParams SampleNoise() {
  EqfImuNoiseParams n;
  n.gyro_noise_density = 0.05;
  n.gyro_random_walk = 0.0002;
  return n;
}

}  // namespace

TEST(EqfAttitude, PredictStationaryHoldsOrientation) {
  EqfAttitudeState x0;
  EqfAttitude est(x0, SmallCov(), SampleNoise(), Eigen::Vector3d(0, 0, -1),
                 Eigen::Vector3d(0.45, 0.05, 0.89));

  for (int i = 0; i < 1000; ++i) {
    est.Predict(Eigen::Vector3d(0, 0, 0), 0.005);
  }

  EXPECT_LT(est.state().R.log().norm(), 1e-9);
}

TEST(EqfAttitude, PredictConstantAngularRateMatchesExpMap) {
  EqfAttitudeState x0;
  EqfAttitude est(x0, SmallCov(), SampleNoise(), Eigen::Vector3d(0, 0, -1),
                 Eigen::Vector3d(0.45, 0.05, 0.89));

  const double dt = 0.001;
  const int steps = 2000;
  for (int i = 0; i < steps; ++i) {
    est.Predict(Eigen::Vector3d(0, 0, 0.5), dt);
  }

  const Sophus::SO3d expected =
      Sophus::SO3d::exp(Eigen::Vector3d(0, 0, 0.5 * steps * dt));
  EXPECT_LT((est.state().R.inverse() * expected).log().norm(), 1e-9);
}

TEST(EqfAttitude, VectorUpdateReducesOrientationError) {
  const Eigen::Vector3d dg(0, 0, -1);
  const Eigen::Vector3d dm(0.45, 0.05, 0.89);

  const Sophus::SO3d R_true = Sophus::SO3d::exp(Eigen::Vector3d(0.2, -0.1, 0.05));
  EqfAttitudeState x0;
  EqfAttitude est(x0, Eigen::Matrix3d::Identity() * 0.1, SampleNoise(), dg, dm);

  eqf_attitude::VectorMeasurement gravity;
  gravity.body_measured = R_true.inverse() * dg;
  gravity.sigma = 0.01;
  eqf_attitude::VectorMeasurement magnetometer;
  magnetometer.body_measured = R_true.inverse() * dm.normalized();
  magnetometer.sigma = 0.01;

  const double err_before = (est.state().R.inverse() * R_true).log().norm();
  est.UpdateVector(gravity, magnetometer, 0.01);
  const double err_after =
      (est.state().R.inverse() * R_true).log().norm();

  EXPECT_LT(err_after, err_before);
  EXPECT_LT(err_after, 0.1);
}

}  // namespace eqf_attitude

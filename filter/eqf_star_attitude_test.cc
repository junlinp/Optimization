#include "eqf_star_attitude.h"

#include <random>

#include "gtest/gtest.h"

namespace eqf_star_attitude {
namespace {

Eigen::Matrix<double, 3, 3> SmallCov() {
  return Eigen::Matrix<double, 3, 3>::Identity() * 1e-6;
}

EqfStarImuNoiseParams SampleNoise() {
  EqfStarImuNoiseParams n;
  n.gyro_noise_density = 0.05;
  n.gyro_random_walk = 0.0002;
  return n;
}

}  // namespace

TEST(EqfStarAttitude, PredictStationaryHoldsOrientation) {
  EqfStarAttitudeState x0;
  EqfStarAttitude est(x0, SmallCov(), SampleNoise(), Eigen::Vector3d(0, 0, -1),
                      Eigen::Vector3d(0.45, 0.05, 0.89));

  for (int i = 0; i < 1000; ++i) {
    est.Predict(Eigen::Vector3d(0, 0, 0), 0.005);
  }

  EXPECT_LT(est.state().R.log().norm(), 1e-9);
}

TEST(EqfStarAttitude, EquivariantOutputBlockMatchesPaperFormula) {
  const Eigen::Vector3d y = Eigen::Vector3d(0.2, -0.1, 0.95).normalized();
  const Eigen::Vector3d y_hat = Eigen::Vector3d(0.15, -0.05, 0.98).normalized();
  const Eigen::Matrix3d R_hat =
      Sophus::SO3d::exp(Eigen::Vector3d(0.1, -0.2, 0.05)).matrix();

  const Eigen::Matrix3d C_star =
      EqfStarAttitude::EquivariantOutputBlock(y, y_hat, R_hat);
  const Eigen::Matrix3d expected =
      0.5 * (Sophus::SO3d::hat(y) + Sophus::SO3d::hat(y_hat));

  EXPECT_LT((C_star - expected).norm(), 1e-12);
}

TEST(EqfStarAttitude, VectorUpdateReducesOrientationError) {
  const Eigen::Vector3d dg(0, 0, -1);
  const Eigen::Vector3d dm(0.45, 0.05, 0.89);

  const Sophus::SO3d R_true = Sophus::SO3d::exp(Eigen::Vector3d(0.2, -0.1, 0.05));
  EqfStarAttitudeState x0;
  EqfStarAttitude est(x0, Eigen::Matrix3d::Identity() * 0.1, SampleNoise(), dg, dm);

  VectorMeasurement gravity;
  gravity.body_measured = R_true.inverse() * dg;
  gravity.sigma = 0.01;
  VectorMeasurement magnetometer;
  magnetometer.body_measured = R_true.inverse() * dm.normalized();
  magnetometer.sigma = 0.01;

  const double err_before = (est.state().R.inverse() * R_true).log().norm();
  est.UpdateVector(gravity, magnetometer, 0.01);
  const double err_after = (est.state().R.inverse() * R_true).log().norm();

  EXPECT_LT(err_after, err_before);
  EXPECT_LT(err_after, 0.1);
}

}  // namespace eqf_star_attitude

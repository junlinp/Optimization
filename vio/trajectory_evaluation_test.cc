#include "trajectory_evaluation.h"

#include <cstdlib>

#include "gtest/gtest.h"
#include "sophus/so3.hpp"

namespace vio {
namespace {

// Bazel's test working directory/runfiles layout isn't guaranteed to be the
// repo root (see euroc_loader_test.cc's FindTestdataMav0Dir), so probe the
// same set of candidates here rather than assume "vio/testdata/mav0"
// resolves directly.
std::vector<std::string> TestdataCandidates() {
  const std::string srcdir = std::getenv("TEST_SRCDIR") ? std::getenv("TEST_SRCDIR") : ".";
  return {
      "vio/testdata/mav0",
      "testdata/mav0",
      srcdir + "/Optimization/vio/testdata/mav0",
      srcdir + "/_main/vio/testdata/mav0",
  };
}

// 10 non-planar, non-collinear points (matches rigid_registration_test.cc's
// SamplePoints style: fixed literals, no RNG).
std::vector<Eigen::Vector3d> SampleGroundTruthPath() {
  return {
      {0, 0, 0},  {1, 0, 0},  {0, 1, 0},  {0, 0, 1},  {1, 1, 0},
      {1, 0, 1},  {0, 1, 1},  {1, 1, 1},  {2, 0.5, 0}, {0.5, 2, 1},
  };
}

}  // namespace

TEST(ComputeAte, ZeroForIdenticalTrajectories) {
  std::vector<TrajectorySample> samples;
  int64_t t = 0;
  for (const auto& p : SampleGroundTruthPath()) {
    samples.push_back({t++, p, p});
  }

  const AteResult ate = ComputeAte(samples);
  EXPECT_EQ(ate.num_samples, 10);
  EXPECT_LT(ate.rmse_m, 1e-9);
  EXPECT_LT(ate.mean_m, 1e-9);
  EXPECT_LT(ate.median_m, 1e-9);
  EXPECT_LT(ate.max_m, 1e-9);
}

// The defining property of ATE vs. a naive raw-position-error metric: a pure
// rigid misalignment between the estimated and ground-truth trajectories
// (as if the estimator ran in a globally rotated/translated frame, while
// tracking shape-perfectly) must be recognized and cancelled by the
// alignment step, leaving zero residual error.
TEST(ComputeAte, RigidMisalignmentIsFullyRecovered) {
  const Eigen::Matrix3d R_true = Sophus::SO3d::exp(Eigen::Vector3d(0.2, -0.3, 0.5)).matrix();
  const Eigen::Vector3d t_true(3, -1, 2);

  std::vector<TrajectorySample> samples;
  int64_t t = 0;
  for (const auto& gt : SampleGroundTruthPath()) {
    // est chosen so that gt == R_true * est + t_true exactly.
    const Eigen::Vector3d est = R_true.transpose() * (gt - t_true);
    samples.push_back({t++, est, gt});
  }

  const AteResult ate = ComputeAte(samples);
  EXPECT_LT(ate.rmse_m, 1e-9);
  EXPECT_LT(ate.max_m, 1e-9);
}

// A per-point-varying (non-rigid) perturbation cannot be fully cancelled by
// a single rigid alignment, so it should show up as a bounded but nonzero
// residual roughly on the order of the injected perturbation.
TEST(ComputeAte, NonRigidPerturbationLeavesBoundedResidual) {
  std::vector<TrajectorySample> samples;
  int64_t t = 0;
  int i = 0;
  for (const auto& gt : SampleGroundTruthPath()) {
    const Eigen::Vector3d perturbation = (i % 2 == 0) ? Eigen::Vector3d(0.1, 0, 0)
                                                      : Eigen::Vector3d(-0.1, 0, 0);
    samples.push_back({t++, gt + perturbation, gt});
    ++i;
  }

  const AteResult ate = ComputeAte(samples);
  EXPECT_GT(ate.rmse_m, 0.01);
  EXPECT_LT(ate.rmse_m, 0.15);
}

TEST(ComputeAte, TooFewSamplesReturnsDefaultResult) {
  std::vector<TrajectorySample> samples = {{0, {0, 0, 0}, {0, 0, 0}}, {1, {1, 0, 0}, {1, 0, 0}}};
  const AteResult ate = ComputeAte(samples);
  EXPECT_EQ(ate.num_samples, 2);
  EXPECT_EQ(ate.rmse_m, 0.0);
}

TEST(FindEurocSequence, ReturnsExplicitPathWhenValid) {
  // Whichever real candidate exists in this environment; explicit_path
  // takes priority over default_candidates regardless of which one it is.
  const std::string real_path = FindEurocSequence("", TestdataCandidates());
  ASSERT_FALSE(real_path.empty());

  const std::string found = FindEurocSequence(real_path, {"some/nonexistent/path"});
  EXPECT_EQ(found, real_path);
}

TEST(FindEurocSequence, FallsBackToDefaultCandidates) {
  std::vector<std::string> candidates = {"some/nonexistent/path"};
  const auto real_candidates = TestdataCandidates();
  candidates.insert(candidates.end(), real_candidates.begin(), real_candidates.end());

  const std::string found = FindEurocSequence("", candidates);
  EXPECT_FALSE(found.empty());
  EXPECT_NE(found, "some/nonexistent/path");
}

TEST(FindEurocSequence, ReturnsEmptyWhenNothingMatches) {
  const std::string found = FindEurocSequence("", {"some/nonexistent/path"});
  EXPECT_EQ(found, "");
}

}  // namespace vio

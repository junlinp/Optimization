// Copyright (c) 2025 junlinp
// All rights reserved.

#include "application/rotation_averaging/rotation_averaging.h"

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cmath>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace {

using Eigen::Quaterniond;
using Eigen::Vector3d;
using Measurements = std::map<std::pair<size_t, size_t>, Quaterniond>;

Quaterniond Rotation(double radians, const Vector3d& axis) {
  return Quaterniond(Eigen::AngleAxisd(radians, axis.normalized()));
}

void ExpectRotation(const Quaterniond& actual, const Quaterniond& expected) {
  EXPECT_NEAR(actual.norm(), 1.0, 1e-10);
  EXPECT_LT(actual.angularDistance(expected), 1e-5);
}

double SquaredError(const Measurements& edges,
                    const std::vector<Quaterniond>& rotations) {
  double sum = 0.0;
  for (const auto& [indices, measurement] : edges) {
    const auto [i, j] = indices;
    const Eigen::Matrix3d predicted =
        rotations[j].toRotationMatrix() * rotations[i].toRotationMatrix().transpose();
    sum += (predicted - measurement.normalized().toRotationMatrix()).squaredNorm();
  }
  return sum;
}

TEST(RotationAveragingTest, RecoversNoncommutingRotationsInFirstNodeGauge) {
  const std::vector<Quaterniond> truth = {
      Rotation(0.8, Vector3d::UnitX()), Rotation(-1.3, Vector3d::UnitY()),
      Rotation(1.8, Vector3d::UnitZ()), Rotation(0.9, Vector3d(1, 2, 3))};
  Measurements edges;
  for (const auto& edge : {std::make_pair(0, 1), {2, 1}, {2, 3}, {3, 0}, {0, 2}}) {
    const auto [i, j] = edge;
    edges[edge] = truth[j] * truth[i].conjugate();
  }
  edges[{2, 1}].coeffs() *= -3.0;

  std::vector<Quaterniond> result;
  ASSERT_TRUE(RotationAveraging::averageRotations(edges, truth.size(), &result));
  ASSERT_EQ(result.size(), truth.size());
  for (size_t i = 0; i < truth.size(); ++i) {
    SCOPED_TRACE(i);
    ExpectRotation(result[i], truth[i] * truth.front().conjugate());
  }
}

TEST(RotationAveragingTest, CertifiesTheChordalMidpointOfConflictingMeasurements) {
  // Expressed in the 0-to-1 direction the measurements have angles 0.2 and 1.0.
  // Their chordal least-squares mean has angle 0.6 and residual angle 0.4 on each edge.
  const Vector3d axis(1, 2, 3);
  const Measurements edges = {{{0, 1}, Rotation(0.2, axis)},
                              {{1, 0}, Rotation(-1.0, axis)}};
  ShonanOptions options;
  ShonanReport report;
  std::vector<Quaterniond> result;
  ASSERT_TRUE(RotationAveraging::averageRotations(edges, 2, &result, options, &report));
  ASSERT_EQ(result.size(), 2u);
  ExpectRotation(result[0], Quaterniond::Identity());
  ExpectRotation(result[1], Rotation(0.6, axis));

  const double optimum = 8.0 * (1.0 - std::cos(0.4));
  const double observed = SquaredError(edges, result);
  EXPECT_NEAR(observed, optimum, 1e-8);
  EXPECT_NEAR(report.squared_error, observed, 1e-9);
  EXPECT_TRUE(report.certified);
  EXPECT_GE(report.optimality_gap, 0.0);
  EXPECT_LE(observed - optimum, report.optimality_gap + 1e-10);
  EXPECT_LT(report.optimality_gap, 1e-5);
  EXPECT_GE(report.maximum_rank_used, 3);
  EXPECT_LE(report.maximum_rank_used, options.max_rank);
}

TEST(RotationAveragingTest, AnchorsEachComponentAndReturnsIdentityForIsolates) {
  const Quaterniond a = Rotation(0.7, Vector3d::UnitX());
  const Quaterniond b = Rotation(-1.2, Vector3d::UnitY());
  const Measurements edges = {{{3, 1}, a}, {{2, 5}, b}};
  std::vector<Quaterniond> result;
  ASSERT_TRUE(RotationAveraging::averageRotations(edges, 6, &result));
  ASSERT_EQ(result.size(), 6u);
  for (size_t i : {0, 1, 2, 4}) {
    SCOPED_TRACE(i);
    ExpectRotation(result[i], Quaterniond::Identity());
  }
  ExpectRotation(result[3], a.conjugate());
  ExpectRotation(result[5], b);
}

TEST(RotationAveragingTest, EmptyGraphProducesOneIdentityPerNode) {
  std::vector<Quaterniond> result = {Rotation(1.0, Vector3d::UnitZ())};
  ASSERT_TRUE(RotationAveraging::averageRotations({}, 0, &result));
  EXPECT_TRUE(result.empty());
  ASSERT_TRUE(RotationAveraging::averageRotations({}, 3, &result));
  ASSERT_EQ(result.size(), 3u);
  for (const auto& rotation : result) {
    ExpectRotation(rotation, Quaterniond::Identity());
  }
}

TEST(RotationAveragingTest, RejectsInvalidMeasurementsAndPreservesOutput) {
  const auto identity = Quaterniond::Identity();
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  const auto infinity = std::numeric_limits<double>::infinity();
  const std::vector<Measurements> invalid = {
      {{{2, 0}, identity}}, {{{0, 2}, identity}},
      {{{0, 1}, Quaterniond(0, 0, 0, 0)}}, {{{0, 1}, Quaterniond(1, nan, 0, 0)}},
      {{{0, 1}, Quaterniond(infinity, 0, 0, 0)}}};
  const Quaterniond sentinel = Rotation(0.8, Vector3d(2, 3, 4));
  for (size_t i = 0; i < invalid.size(); ++i) {
    SCOPED_TRACE(i);
    std::vector<Quaterniond> result = {sentinel};
    EXPECT_FALSE(RotationAveraging::averageRotations(invalid[i], 2, &result));
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().coeffs(), sentinel.coeffs());
  }
}

TEST(RotationAveragingTest, EscapesAnUncertifiedRankThreeMinimum) {
  // The identity spanning tree puts cyclic SO(3) minimization in a winding
  // configuration. The two conflicting chords make that minimum suboptimal.
  const auto id = Quaterniond::Identity();
  const double radians = std::acos(-1.0) / 180.0;
  const Measurements edges = {{{0, 1}, id}, {{0, 2}, id}, {{0, 3}, id},
                              {{1, 3}, Rotation(60 * radians, Vector3d::UnitZ())},
                              {{2, 3}, Rotation(-165 * radians, Vector3d::UnitZ())}};
  ShonanOptions options;
  options.max_rank = 3;
  ShonanReport report;
  std::vector<Quaterniond> result{id};
  ASSERT_TRUE(RotationAveraging::averageRotations(edges, 4, &result, options, &report));
  ASSERT_EQ(result.size(), 4u);
  EXPECT_FALSE(report.certified);
  EXPECT_EQ(report.maximum_rank_used, 3);
  EXPECT_NEAR(report.squared_error, 7.70591677765, 1e-7);
  EXPECT_GE(report.optimality_gap, report.squared_error - 6.93286339577 - 1e-9);

  options.max_rank = 6;
  ASSERT_TRUE(RotationAveraging::averageRotations(edges, 4, &result, options, &report));
  ASSERT_EQ(result.size(), 4u);
  EXPECT_TRUE(report.certified);
  EXPECT_GT(report.maximum_rank_used, 3);
  EXPECT_NEAR(SquaredError(edges, result), 6.93286339577, 1e-7);
  EXPECT_LE(report.optimality_gap, options.gap_tolerance * edges.size());
}

TEST(RotationAveragingTest, SelfMeasurementsContributeOnlyConstantCost) {
  Measurements edges = {{{0, 0}, Rotation(0.7, Vector3d::UnitY())}};
  ShonanReport report;
  std::vector<Quaterniond> result;
  ASSERT_TRUE(RotationAveraging::averageRotations(edges, 1, &result, ShonanOptions{}, &report));
  ASSERT_EQ(result.size(), 1u);
  ExpectRotation(result[0], Quaterniond::Identity());
  EXPECT_NEAR(report.squared_error, SquaredError(edges, result), 1e-12);
  EXPECT_DOUBLE_EQ(report.optimality_gap, 0.0);
}

TEST(RotationAveragingTest, DistributesLoopClosureErrorOnALongSparseCycle) {
  constexpr size_t n = 256;
  const Vector3d axis(1.0, 2.0, -0.5);
  Measurements edges;
  for (size_t i = 0; i + 1 < n; ++i) edges[{i, i + 1}] = Quaterniond::Identity();
  edges[{0, n - 1}] = Rotation(1.0, axis);

  ShonanReport report;
  std::vector<Quaterniond> result;
  ASSERT_TRUE(RotationAveraging::averageRotations(edges, n, &result, ShonanOptions{}, &report));
  ASSERT_EQ(result.size(), n);
  EXPECT_TRUE(report.certified);
  // Each of the n residual angles at the optimum has magnitude 1/n.
  const double optimum = 8.0 * n * std::pow(std::sin(0.5 / n), 2);
  EXPECT_NEAR(SquaredError(edges, result), optimum, 1e-8);
  ExpectRotation(result[n / 2], Rotation(0.5, axis));
}

TEST(RotationAveragingTest, CertifiesSparseGraphAfterRankLift) {
  constexpr size_t n = 80;
  const double radians = std::acos(-1.0) / 180.0;
  const auto id = Quaterniond::Identity();
  Measurements edges = {{{0, 1}, id}, {{0, 2}, id}, {{0, 3}, id},
                        {{1, 3}, Rotation(120 * radians, Vector3d::UnitZ())},
                        {{2, 3}, Rotation(-170 * radians,
                                    Vector3d(std::sin(15 * radians), 0, std::cos(15 * radians)))}};
  // These leaves increase the certificate dimension without changing the
  // optimum of the four-vertex core. Every leaf can match its parent.
  for (size_t i = 4; i < n; ++i) edges[{0, i}] = id;

  ShonanReport report;
  std::vector<Quaterniond> result;
  ASSERT_TRUE(RotationAveraging::averageRotations(edges, n, &result, ShonanOptions{}, &report));
  ASSERT_EQ(result.size(), n);
  EXPECT_TRUE(report.certified);
  EXPECT_GT(report.maximum_rank_used, 3);
  EXPECT_NEAR(SquaredError(edges, result), 7.21950345465, 1e-7);
}

TEST(RotationAveragingTest, ReportsAnUncertifiedEstimateForANontightRelaxation) {
  // Three two-edge paths have mutually incompatible half-turn measurements.
  // For terminal quaternion (w,x,y,z), optimizing the intermediate rotations
  // gives cost 24 - 8 (|x| + |y| + |z|), minimized at 24 - 8 sqrt(3).
  Measurements edges;
  for (size_t axis = 0; axis < 3; ++axis) {
    edges[{0, 2 + axis}] = Quaterniond::Identity();
    edges[{2 + axis, 1}] = Quaterniond(0, axis == 0, axis == 1, axis == 2);
  }
  std::vector<Quaterniond> result;
  ShonanReport report;
  ASSERT_TRUE(RotationAveraging::averageRotations(edges, 5, &result, ShonanOptions{}, &report));
  ASSERT_EQ(result.size(), 5u);
  const double optimum = 24.0 - 8.0 * std::sqrt(3.0);
  EXPECT_FALSE(report.certified);
  EXPECT_NEAR(report.squared_error, SquaredError(edges, result), 1e-9);
  EXPECT_GE(report.squared_error, optimum - 1e-9);
  EXPECT_GE(report.optimality_gap, report.squared_error - optimum - 1e-9);
  EXPECT_GT(report.optimality_gap, 1e-3);
}

TEST(RotationAveragingTest, RejectsInvalidOptionsAndNullOutput) {
  const Measurements edges = {{{0, 1}, Quaterniond::Identity()}};
  std::vector<Quaterniond> result;
  ShonanReport report;
  EXPECT_FALSE(RotationAveraging::averageRotations(edges, 2, nullptr));
  ShonanOptions options;
  options.max_rank = 2;
  EXPECT_FALSE(RotationAveraging::averageRotations(edges, 2, &result, options, &report));
  options = ShonanOptions{};
  options.max_sweeps = 0;
  EXPECT_FALSE(RotationAveraging::averageRotations(edges, 2, &result, options, &report));
  options = ShonanOptions{};
  options.gap_tolerance = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(RotationAveraging::averageRotations(edges, 2, &result, options, &report));
}

}  // namespace

#include "gtest/gtest.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "linear_programing.h"
#include "mps_problem.h"

namespace {

std::string MpsPath(const std::string& filename) {
  const std::vector<std::string> candidates = {
      "programing/testdata/" + filename,
      std::string(std::getenv("TEST_SRCDIR") ? std::getenv("TEST_SRCDIR") : ".") +
          "/Optimization/programing/testdata/" + filename,
      std::string(std::getenv("TEST_SRCDIR") ? std::getenv("TEST_SRCDIR") : ".") +
          "/_main/programing/testdata/" + filename,
  };
  for (const auto& path : candidates) {
    std::ifstream in(path);
    if (in.good()) {
      return path;
    }
  }
  return candidates.front();
}

struct MpsBenchmarkCase {
  const char* filename;
  double expected_objective;
  double objective_tol;
  double residual_tol;
};

class MpsBenchmarkTest : public ::testing::TestWithParam<MpsBenchmarkCase> {};

TEST_P(MpsBenchmarkTest, SolveWithLPSolver2) {
  const MpsBenchmarkCase& test_case = GetParam();
  const std::string path = MpsPath(test_case.filename);
  MPSProblem prob;
  ASSERT_NO_THROW(prob = read_mps(path)) << "Failed to open " << path;
  ASSERT_FALSE(prob.col_index.empty());
  ASSERT_FALSE(prob.row_index.empty());

  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  Eigen::VectorXd c;
  BuildDenseLp(prob, &A, &b, &c);

  Eigen::VectorXd x = Eigen::VectorXd::Ones(c.size());
  const auto start = std::chrono::steady_clock::now();
  LPSolver2(c, A, b, x);
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start)
          .count();

  const double residual = (A * x - b).norm();
  const double objective = c.dot(x);
  std::cout << test_case.filename << " rows=" << A.rows()
            << " cols=" << A.cols() << " time_ms=" << elapsed_ms
            << " obj=" << objective << " ||Ax-b||=" << residual << std::endl;

  EXPECT_LT(residual, test_case.residual_tol);
  EXPECT_NEAR(objective, test_case.expected_objective, test_case.objective_tol);
  EXPECT_GE(x.minCoeff(), -1e-6);
}

INSTANTIATE_TEST_SUITE_P(
    Fixtures, MpsBenchmarkTest,
    ::testing::Values(
        MpsBenchmarkCase{"lp_afiro_style.mps", -0.125, 1e-5, 1e-5},
        MpsBenchmarkCase{"lp_equal_split.mps", 6.0, 1e-6, 1e-6},
        MpsBenchmarkCase{"lp_blend.mps", -36.0, 1e-6, 1e-6}));

class NetlibBenchmarkTest : public ::testing::TestWithParam<MpsBenchmarkCase> {};

TEST(MpsReaderTest, ReadsUnnamedRhsInBlend) {
  const MPSProblem prob = read_mps(MpsPath("external/netlib/blend.mps"));
  const double values[] = {23.26, 5.25, 26.32, 21.05, 13.45, 2.58, 10., 10.};
  ASSERT_EQ(prob.rhs.size(), 8u);
  for (int i = 0; i < 8; ++i) {
    const auto row = prob.row_index.at(std::to_string(65 + i));
    EXPECT_DOUBLE_EQ(prob.rhs.at(row), values[i]);
  }
}

TEST(MpsReaderTest, ReadsNamedAndUnnamedRhsAndRanges) {
  for (const char* name : {"lp_named_rhs_ranges.mps", "lp_unnamed_rhs_ranges.mps"}) {
    const MPSProblem prob = read_mps(MpsPath(name));
    ASSERT_EQ(prob.rhs.size(), 2u) << name;
    ASSERT_EQ(prob.ranges.size(), 2u) << name;
    EXPECT_DOUBLE_EQ(prob.rhs.at(prob.row_index.at("ROW1")), 3.0);
    EXPECT_DOUBLE_EQ(prob.rhs.at(prob.row_index.at("ROW2")), 2.0);
    EXPECT_DOUBLE_EQ(prob.ranges.at(prob.row_index.at("ROW1")), 1.0);
    EXPECT_DOUBLE_EQ(prob.ranges.at(prob.row_index.at("ROW2")), 0.5);
  }
}

TEST(LPSolver3Test, IterationLimitDoesNotClaimOptimality) {
  Eigen::MatrixXd dense(1, 2); dense << 1, 1;
  Eigen::SparseMatrix<double> A = dense.sparseView();
  Eigen::VectorXd b(1), c(2); b << 10; c << 5, 6;
  const LPResult result = SolveLP3(c, A, b, 0);
  EXPECT_EQ(result.status, LPStatus::IterationLimit);
  EXPECT_EQ(result.iterations, 0u);
  EXPECT_NEAR(result.primal_residual, (A * result.x - b).norm(), 1e-12);
  EXPECT_NEAR(result.dual_residual, (c - A.transpose() * result.y - result.s).norm(), 1e-12);
  EXPECT_NEAR(result.gap, result.x.dot(result.s), 1e-12);
}

TEST(LPSolver3Test, HandlesNoEqualityConstraints) {
  Eigen::SparseMatrix<double> A(0, 1);
  Eigen::VectorXd b(0), c(1); c << 1;
  const LPResult solved = SolveLP3(c, A, b);
  ASSERT_EQ(solved.status, LPStatus::Optimal);
  EXPECT_NEAR(solved.x(0), 0.0, 1e-6);
  EXPECT_LE(solved.dual_residual, 1e-6 * (1.0 + c.norm()));
  c(0) = -1;
  EXPECT_NE(SolveLP3(c, A, b, 30).status, LPStatus::Optimal);
}

TEST_P(NetlibBenchmarkTest, SolveWithLPSolver3) {
  const MpsBenchmarkCase& test_case = GetParam();
  const std::string path = MpsPath(test_case.filename);
  MPSProblem prob;
  ASSERT_NO_THROW(prob = read_mps(path)) << "Failed to open " << path;
  ASSERT_FALSE(prob.col_index.empty());
  ASSERT_FALSE(prob.row_index.empty());

  const StandardFormLp lp = BuildStandardFormLp(prob);
  ASSERT_GT(lp.A.rows(), 0);
  ASSERT_GT(lp.A.cols(), 0);

  const auto start = std::chrono::steady_clock::now();
  const LPResult solved = SolveLP3(lp.c, lp.A, lp.b);
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start)
          .count();

  ASSERT_EQ(solved.status, LPStatus::Optimal) << "iterations=" << solved.iterations
      << " primal=" << solved.primal_residual << " dual=" << solved.dual_residual
      << " gap=" << solved.gap;
  const Eigen::VectorXd& x = solved.x;
  const double residual = (lp.A * x - lp.b).norm();
  const double objective = lp.c.dot(x) + lp.objective_offset;
  const double rel_scale = std::max(1.0, std::abs(test_case.expected_objective));
  std::ostringstream result;
  result << std::setprecision(12) << test_case.filename
         << " orig_rows=" << prob.row_index.size()
         << " orig_cols=" << prob.col_index.size()
         << " std_rows=" << lp.A.rows() << " std_cols=" << lp.A.cols()
         << " nnz=" << lp.A.nonZeros()
         << " target_obj=" << test_case.expected_objective
         << " computed_obj=" << objective << " time_ms=" << elapsed_ms
         << " ||Ax-b||=" << residual << " dual_residual=" << solved.dual_residual
         << " gap=" << solved.gap << " iterations=" << solved.iterations;
  std::cout << result.str() << std::endl;

  EXPECT_LT(residual, test_case.residual_tol);
  EXPECT_NEAR(objective, test_case.expected_objective,
              test_case.objective_tol * rel_scale);
  EXPECT_GE(x.minCoeff(), -1e-6);
  EXPECT_GE(solved.s.minCoeff(), 0.0);
  EXPECT_LE((lp.c - lp.A.transpose() * solved.y - solved.s).norm(),
            1e-6 * (1.0 + lp.c.norm()));
  EXPECT_LE(solved.x.dot(solved.s),
            1e-6 * (1.0 + std::abs(lp.c.dot(solved.x)) + std::abs(lp.b.dot(solved.y))));
}

INSTANTIATE_TEST_SUITE_P(
    Netlib, NetlibBenchmarkTest,
    ::testing::Values(
        MpsBenchmarkCase{"netlib/afiro.mps", -4.6475314286e2, 1e-4, 1e-4},
        MpsBenchmarkCase{"netlib/adlittle.mps", 2.2549496316e5, 1e-4, 1e-4},
        MpsBenchmarkCase{"netlib/share2b.mps", -4.1573224074e2, 1e-4, 1e-4},
        MpsBenchmarkCase{"external/netlib/sc50a.mps", -64.5750770586, 1e-4, 1e-4},
        MpsBenchmarkCase{"external/netlib/sc50b.mps", -70.0, 1e-4, 1e-4},
        MpsBenchmarkCase{"external/netlib/blend.mps", -30.812149846, 1e-4, 1e-4}));

}  // namespace

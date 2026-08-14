#include "gtest/gtest.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
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

}  // namespace

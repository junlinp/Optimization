#include "jordan_conic.h"
#include <gtest/gtest.h>
#include <cmath>

using jordan::ConeBlock;
using jordan::ConeType;

TEST(JordanBackend, SelectsBackendAndRejectsUnsupportedCones) {
  Eigen::MatrixXd A(1, 2); A << 1, 1;
  Eigen::VectorXd b(1), c(2); b << 1; c << 1, 2;
  const std::vector<ConeBlock> lp = {{ConeType::Nonnegative, 2}};
  const auto automatic = jordan::Solve(A, b, c, lp);
  ASSERT_EQ(automatic.status, jordan::Status::Optimal);
  EXPECT_EQ(automatic.backend, jordan::Backend::SparseLP);
  jordan::Options options; options.backend = jordan::Backend::DenseJordan;
  const auto dense = jordan::Solve(A, b, c, lp, options);
  ASSERT_EQ(dense.status, jordan::Status::Optimal);
  EXPECT_EQ(dense.backend, jordan::Backend::DenseJordan);
  EXPECT_NEAR(c.dot(automatic.x), c.dot(dense.x), 1e-6);
  options.backend = jordan::Backend::SparseLP;
  EXPECT_EQ(jordan::Solve(A, b, c, {{ConeType::Lorentz, 2}}, options).status,
            jordan::Status::InvalidInput);
  Eigen::SparseMatrix<double> sparse = A.sparseView();
  const auto from_sparse = jordan::SolveSparse(sparse, b, c, lp);
  ASSERT_EQ(from_sparse.status, jordan::Status::Optimal);
  EXPECT_EQ(from_sparse.backend, jordan::Backend::SparseLP);
}

TEST(JordanBackend, SparseAndDensePerformSamePredictorCorrectorIteration) {
  Eigen::MatrixXd A(2, 3); A << 1, 1, 0, 0, 1, 1;
  Eigen::VectorXd b(2), c(3); b << 2, 3; c << 1, 2, 3;
  const std::vector<ConeBlock> cones = {{ConeType::Nonnegative, 3}};
  jordan::Options options; options.max_iterations = 1; options.tolerance = 1e-14;
  options.backend = jordan::Backend::SparseLP;
  const auto sparse = jordan::Solve(A, b, c, cones, options);
  options.backend = jordan::Backend::DenseJordan;
  const auto dense = jordan::Solve(A, b, c, cones, options);
  ASSERT_EQ(sparse.status, jordan::Status::MaxIterations);
  ASSERT_EQ(dense.status, jordan::Status::MaxIterations);
  EXPECT_TRUE(sparse.x.isApprox(dense.x, 1e-9));
  EXPECT_TRUE(sparse.y.isApprox(dense.y, 1e-9));
  EXPECT_TRUE(sparse.s.isApprox(dense.s, 1e-9));
}

TEST(JordanAlgebra, ProductIdentitiesAndLorentz) {
  std::vector<ConeBlock> cones = {{ConeType::Nonnegative, 2}, {ConeType::Lorentz, 3}, {ConeType::Semidefinite, 2}};
  Eigen::VectorXd x(8); x << 2, 3, 4, 1, 2, 3, 4, std::sqrt(2.0);
  EXPECT_TRUE(jordan::Product(cones, x, jordan::Identity(cones)).isApprox(x));
  auto z = jordan::Product(cones, x, x);
  EXPECT_NEAR(z(2), 21, 1e-12);
  EXPECT_NEAR(z(3), 8, 1e-12);
  EXPECT_NEAR(z(5), 10, 1e-12);
  EXPECT_NEAR(z(6), 17, 1e-12);
  EXPECT_NEAR(z(7), 7 * std::sqrt(2.0), 1e-12);
}

TEST(JordanSolver, LinearProgram) {
  Eigen::MatrixXd A(1, 2); A << 1, 1;
  Eigen::VectorXd b(1), c(2); b << 1; c << 1, 2;
  auto r = jordan::Solve(A, b, c, {{ConeType::Nonnegative, 2}});
  ASSERT_EQ(r.status, jordan::Status::Optimal);
  EXPECT_NEAR(c.dot(r.x), 1, 1e-6);
  EXPECT_LT(r.primal_residual, 1e-7); EXPECT_LT(r.dual_residual, 1e-7); EXPECT_LT(r.gap, 1e-7);
}

TEST(JordanSolver, SecondOrderCone) {
  // min t subject to (t, 3, 4) in the Lorentz cone; optimum t=5.
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2, 3); A(0,1)=1; A(1,2)=1;
  Eigen::VectorXd b(2), c(3); b << 3, 4; c << 1, 0, 0;
  auto r = jordan::Solve(A, b, c, {{ConeType::Lorentz, 3}});
  ASSERT_EQ(r.status, jordan::Status::Optimal);
  EXPECT_NEAR(c.dot(r.x), 5, 1e-6);
  EXPECT_GE(r.x(0) + 1e-8, r.x.tail(2).norm());
  EXPECT_LT(r.primal_residual, 1e-7); EXPECT_LT(r.dual_residual, 1e-7); EXPECT_LT(r.gap, 1e-7);
}

TEST(JordanSolver, SemidefiniteCone) {
  // min trace(X), X_01=1, X positive semidefinite; optimum 2.
  Eigen::MatrixXd A(1,3); A << 0, 0, 1/std::sqrt(2.0);
  Eigen::VectorXd b(1), c(3); b << 1; c << 1, 1, 0;
  auto r = jordan::Solve(A, b, c, {{ConeType::Semidefinite, 2}});
  ASSERT_EQ(r.status, jordan::Status::Optimal);
  EXPECT_NEAR(c.dot(r.x), 2, 1e-6);
  EXPECT_GE(r.x(0)*r.x(1)-r.x(2)*r.x(2)/2, -1e-8);
  EXPECT_LT(r.primal_residual, 1e-7); EXPECT_LT(r.dual_residual, 1e-7); EXPECT_LT(r.gap, 1e-7);
}

TEST(JordanSolver, MixedConesAndInvalidInputs) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, 5);
  A(0,0)=1; A(1,2)=1; A(2,4)=1;
  Eigen::VectorXd b(3), c = Eigen::VectorXd::Zero(5); b << 2, 3, 4; c(0)=1; c(1)=1; c(3)=1;
  auto r = jordan::Solve(A, b, c, {{ConeType::Nonnegative,1}, {ConeType::Lorentz,2}, {ConeType::Lorentz,2}});
  ASSERT_EQ(r.status, jordan::Status::Optimal);
  EXPECT_NEAR(c.dot(r.x), 9, 1e-6);
  EXPECT_EQ(jordan::Solve(A, b, c, {{ConeType::Lorentz,1}}).status, jordan::Status::InvalidInput);
  EXPECT_EQ(jordan::Solve(A, b, c, {{ConeType::Nonnegative,5}}, {-1, 1e-8}).status, jordan::Status::InvalidInput);
}

TEST(JordanSolver, MixedSemidefiniteAndOrthant) {
  // min trace(X) + z, X_01=1, z=2, X PSD and z nonnegative.
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2, 4);
  A(0, 2) = 1/std::sqrt(2.0); A(1, 3) = 1;
  Eigen::VectorXd b(2), c(4); b << 1, 2; c << 1, 1, 0, 1;
  auto r = jordan::Solve(A, b, c, {{ConeType::Semidefinite,2}, {ConeType::Nonnegative,1}});
  ASSERT_EQ(r.status, jordan::Status::Optimal);
  EXPECT_NEAR(c.dot(r.x), 4, 1e-6);
  EXPECT_LT(r.primal_residual, 1e-7); EXPECT_LT(r.dual_residual, 1e-7); EXPECT_LT(r.gap, 1e-7);
}

TEST(JordanSolver, ReportsIterationLimit) {
  Eigen::MatrixXd A(1, 2); A << 1, 1;
  Eigen::VectorXd b(1), c(2); b << 1; c << 1, 2;
  auto r = jordan::Solve(A, b, c, {{ConeType::Nonnegative,2}}, {1, 1e-12});
  EXPECT_EQ(r.status, jordan::Status::MaxIterations);
  EXPECT_EQ(r.iterations, 1);
}

TEST(JordanDifficult, PoorlyScaledLinearConstraints) {
  // The equality fixes x1=1 and x2=2 despite six orders of row scaling.
  Eigen::MatrixXd A(2, 2); A << 1e6,0, 0,1e-4;
  Eigen::VectorXd b(2), c(2); b << 1e6,2e-4; c << 2,3;
  const auto r=jordan::Solve(A,b,c,{{ConeType::Nonnegative,2}});
  ASSERT_EQ(r.status,jordan::Status::Optimal);
  EXPECT_NEAR(c.dot(r.x),8,1e-6);
  EXPECT_LE((A*r.x-b).norm(), 1e-8*(1+b.norm()));
  EXPECT_LE((c-A.transpose()*r.y-r.s).norm(), 1e-8*(1+c.norm()));
}

TEST(JordanDifficult, NearlySingularSemidefiniteOptimum) {
  // diag(X)=(1,1e-6), maximize X01; optimum X01=0.001.
  Eigen::MatrixXd A(2,3); A << 1,0,0, 0,1,0;
  Eigen::VectorXd b(2), c(3); b << 1,1e-6; c << 0,0,-1/std::sqrt(2.0);
  const auto r=jordan::Solve(A,b,c,{{ConeType::Semidefinite,2}});
  ASSERT_EQ(r.status,jordan::Status::Optimal);
  EXPECT_NEAR(c.dot(r.x),-0.001,1e-6);
  EXPECT_GE(r.x(0)*r.x(1)-r.x(2)*r.x(2)/2,-1e-9);
  EXPECT_LE((A*r.x-b).norm(),1e-8*(1+b.norm()));
  EXPECT_LE((c-A.transpose()*r.y-r.s).norm(),1e-8*(1+c.norm()));
}

TEST(JordanDifficult, DependentEqualitiesCannotClaimFalseOptimum) {
  // Feasible, but duplicate equalities make the current KKT system singular.
  Eigen::MatrixXd A(2,2); A << 1,1, 2,2;
  Eigen::VectorXd b(2), c(2); b << 1,2; c << 1,2;
  const auto r=jordan::Solve(A,b,c,{{ConeType::Nonnegative,2}});
  if (r.status == jordan::Status::Optimal) {
    EXPECT_NEAR(c.dot(r.x),1,1e-6);
    EXPECT_LE((A*r.x-b).norm(),1e-8*(1+b.norm()));
    EXPECT_LE((c-A.transpose()*r.y-r.s).norm(),1e-8*(1+c.norm()));
    EXPECT_LE(r.gap,1e-8*(1+std::abs(c.dot(r.x))));
  } else {
    EXPECT_EQ(r.status,jordan::Status::NumericalFailure);
  }
  std::cout << "dependent equalities status=" << static_cast<int>(r.status) << '\n';
}

TEST(JordanDifficult, InfeasibleCannotClaimOptimal) {
  // x=-1 and x>=0 are incompatible. No infeasibility certificate is implemented.
  Eigen::MatrixXd A(1,1); A << 1;
  Eigen::VectorXd b(1), c(1); b << -1; c << 1;
  const auto r=jordan::Solve(A,b,c,{{ConeType::Nonnegative,1}});
  EXPECT_NE(r.status,jordan::Status::Optimal);
  std::cout << "infeasible status=" << static_cast<int>(r.status) << '\n';
}

TEST(JordanDifficult, UnboundedCannotClaimOptimal) {
  // min -x, x>=0, with no equalities. No unboundedness certificate exists yet.
  Eigen::MatrixXd A(0,1);
  Eigen::VectorXd b(0), c(1); c << -1;
  const auto r=jordan::Solve(A,b,c,{{ConeType::Nonnegative,1}});
  EXPECT_NE(r.status,jordan::Status::Optimal);
  std::cout << "unbounded status=" << static_cast<int>(r.status) << '\n';
}

TEST(JordanDifficult, InfeasibleSecondOrderConeCannotClaimOptimal) {
  // (0, 1) is outside the Lorentz cone.
  Eigen::MatrixXd A=Eigen::MatrixXd::Identity(2,2);
  Eigen::VectorXd b(2), c=Eigen::VectorXd::Zero(2); b << 0,1;
  const auto r=jordan::Solve(A,b,c,{{ConeType::Lorentz,2}});
  EXPECT_NE(r.status,jordan::Status::Optimal);
  std::cout << "infeasible SOCP status=" << static_cast<int>(r.status) << '\n';
}

TEST(JordanDifficult, InfeasibleSemidefiniteCannotClaimOptimal) {
  // X_00=-1 is impossible for a positive semidefinite matrix.
  Eigen::MatrixXd A(1,3); A << 1,0,0;
  Eigen::VectorXd b(1), c=Eigen::VectorXd::Zero(3); b << -1;
  const auto r=jordan::Solve(A,b,c,{{ConeType::Semidefinite,2}});
  EXPECT_NE(r.status,jordan::Status::Optimal);
  std::cout << "infeasible SDP status=" << static_cast<int>(r.status) << '\n';
}

TEST(JordanDifficult, CoupledThreeConeTypes) {
  // min z+t+trace(X); z=2, Lorentz tail=3, X01=1: optimum 7.
  Eigen::MatrixXd A=Eigen::MatrixXd::Zero(3,6);
  A(0,0)=1; A(1,2)=1; A(2,5)=1/std::sqrt(2.0);
  Eigen::VectorXd b(3),c(6); b << 2,3,1; c << 1,1,0,1,1,0;
  const auto r=jordan::Solve(A,b,c,{{ConeType::Nonnegative,1},
                                  {ConeType::Lorentz,2},
                                  {ConeType::Semidefinite,2}});
  ASSERT_EQ(r.status,jordan::Status::Optimal);
  EXPECT_NEAR(c.dot(r.x),7,1e-6);
  EXPECT_LE((A*r.x-b).norm(),1e-8*(1+b.norm()));
  EXPECT_LE((c-A.transpose()*r.y-r.s).norm(),1e-8*(1+c.norm()));
}

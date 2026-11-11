#include "jordan_conic.h"

#include <gtest/gtest.h>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
using jordan::ConeBlock;
using jordan::ConeType;
struct Case {
  std::string name;
  Eigen::MatrixXd A;
  Eigen::VectorXd b, c;
  std::vector<ConeBlock> cones;
  double optimum;
};

std::vector<Case> Cases() {
  std::vector<Case> cases;
  {
    // min x1 + 2x2 + 4x3, sum(x)=3, x>=0.
    Eigen::MatrixXd A(1,3); A << 1,1,1;
    Eigen::VectorXd b(1), c(3); b << 3; c << 1,2,4;
    cases.push_back({"lp_simplex", A,b,c,{{ConeType::Nonnegative,3}},3});
  }
  {
    // max 3x + 5y subject to x<=3, 2y<=12, 3x+2y<=18.
    Eigen::MatrixXd A(3,5); A << 1,0,1,0,0, 0,2,0,1,0, 3,2,0,0,1;
    Eigen::VectorXd b(3), c(5); b << 3,12,18; c << -3,-5,0,0,0;
    cases.push_back({"lp_inequalities", A,b,c,{{ConeType::Nonnegative,5}},-36});
  }
  {
    // Equality constraints force x1=x2 and x2+x3=2; min x3.
    Eigen::MatrixXd A(2,3); A << 1,-1,0, 0,1,1;
    Eigen::VectorXd b(2), c(3); b << 0,2; c << 0,0,1;
    cases.push_back({"lp_coupled_equalities", A,b,c,{{ConeType::Nonnegative,3}},0});
  }
  {
    // min t with (t, 1, 2, 2) in a Lorentz cone.
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3,4);
    A(0,1)=1; A(1,2)=1; A(2,3)=1;
    Eigen::VectorXd b(3), c(4); b << 1,2,2; c << 1,0,0,0;
    cases.push_back({"socp_norm_3d", A,b,c,{{ConeType::Lorentz,4}},3});
  }
  {
    // min t1+t2 with (t1,3,4), (t2,5,12) in Lorentz cones.
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(4,6);
    A(0,1)=1; A(1,2)=1; A(2,4)=1; A(3,5)=1;
    Eigen::VectorXd b(4), c(6); b << 3,4,5,12; c << 1,0,0,1,0,0;
    cases.push_back({"socp_two_blocks", A,b,c,{{ConeType::Lorentz,3},{ConeType::Lorentz,3}},18});
  }
  {
    // min t1+t2, u1+u2=3, v1+v2=4. Triangle inequality gives 5.
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2,6);
    A(0,1)=1; A(0,4)=1; A(1,2)=1; A(1,5)=1;
    Eigen::VectorXd b(2), c(6); b << 3,4; c << 1,0,0,1,0,0;
    cases.push_back({"socp_coupled_blocks", A,b,c,{{ConeType::Lorentz,3},{ConeType::Lorentz,3}},5});
  }
  {
    // PSD matrix diag=(1,4): max X01=2.
    Eigen::MatrixXd A(2,3); A << 1,0,0, 0,1,0;
    Eigen::VectorXd b(2), c(3); b << 1,4; c << 0,0,-1/std::sqrt(2.0);
    cases.push_back({"sdp_fixed_diagonal", A,b,c,{{ConeType::Semidefinite,2}},-2});
  }
  {
    // Correlation matrix diag=(1,1,1): max X01+X02+X12=3.
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3,6); A.block(0,0,3,3).setIdentity();
    Eigen::VectorXd b=Eigen::VectorXd::Ones(3), c(6);
    c << 0,0,0,-1/std::sqrt(2.0),-1/std::sqrt(2.0),-1/std::sqrt(2.0);
    cases.push_back({"sdp_correlation_3x3", A,b,c,{{ConeType::Semidefinite,3}},-3});
  }
  {
    // min trace(X), X01=1, X02=2; optimum 2*sqrt(5).
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2,6);
    A(0,3)=1/std::sqrt(2.0); A(1,4)=1/std::sqrt(2.0);
    Eigen::VectorXd b(2), c(6); b << 1,2; c << 1,1,1,0,0,0;
    cases.push_back({"sdp_trace_3x3", A,b,c,{{ConeType::Semidefinite,3}},2*std::sqrt(5.0)});
  }
  return cases;
}

void CheckCone(const Eigen::VectorXd& v, const std::vector<ConeBlock>& cones) {
  int offset=0;
  for (const auto& cone : cones) {
    if (cone.type == ConeType::Nonnegative) {
      EXPECT_GE(v.segment(offset,cone.size).minCoeff(),-1e-7);
      offset += cone.size;
    } else if (cone.type == ConeType::Lorentz) {
      EXPECT_GE(v(offset)+1e-7,v.segment(offset+1,cone.size-1).norm());
      offset += cone.size;
    } else {
      Eigen::MatrixXd M = Eigen::MatrixXd::Zero(cone.size,cone.size);
      for (int i=0;i<cone.size;++i) M(i,i)=v(offset+i);
      int k=offset+cone.size;
      for (int j=1;j<cone.size;++j) for (int i=0;i<j;++i)
        M(i,j)=M(j,i)=v(k++)/std::sqrt(2.0);
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(M);
      ASSERT_EQ(eig.info(), Eigen::Success);
      EXPECT_GE(eig.eigenvalues().minCoeff(),-1e-7);
      offset=k;
    }
  }
}

class JordanDataset : public ::testing::TestWithParam<Case> {};
TEST_P(JordanDataset, MatchesAnalyticalOptimumAndKkt) {
  const auto& p=GetParam();
  const auto result=jordan::Solve(p.A,p.b,p.c,p.cones);
  std::cout << p.name << " target=" << p.optimum << " objective="
            << (result.x.size() ? p.c.dot(result.x) : 0) << " iterations="
            << result.iterations << " primal=" << result.primal_residual
            << " dual=" << result.dual_residual << " gap=" << result.gap << '\n';
  ASSERT_EQ(result.status,jordan::Status::Optimal) << p.name;
  EXPECT_NEAR(p.c.dot(result.x),p.optimum,1e-6);
  EXPECT_LT((p.A*result.x-p.b).norm(),1e-7);
  EXPECT_LT((p.c-p.A.transpose()*result.y-result.s).norm(),1e-7);
  EXPECT_GE(result.x.dot(result.s), -1e-10);
  EXPECT_LE(result.x.dot(result.s), 1e-8 * (1 + std::abs(p.c.dot(result.x))));
  CheckCone(result.x,p.cones);
  CheckCone(result.s,p.cones);
}
INSTANTIATE_TEST_SUITE_P(Analytical, JordanDataset, ::testing::ValuesIn(Cases()),
                         [](const auto& info) { return info.param.name; });
}  // namespace

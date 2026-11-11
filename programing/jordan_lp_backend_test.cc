#include "jordan_conic.h"
#include "mps_problem.h"
#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace {
struct Case { const char* file; double objective; };
class BackendComparison : public ::testing::TestWithParam<Case> {};
TEST_P(BackendComparison, SameNetlibProblem) {
  const char* dir=std::getenv("TEST_SRCDIR");
  const char* workspace=std::getenv("TEST_WORKSPACE");
  ASSERT_NE(dir,nullptr); ASSERT_NE(workspace,nullptr);
  const std::string path=std::string(dir)+"/"+workspace+"/programing/testdata/"+GetParam().file;
  const auto lp=BuildStandardFormLp(read_mps(path));
  std::vector<jordan::ConeBlock> cones={{jordan::ConeType::Nonnegative,static_cast<int>(lp.c.size())}};
  double values[2] = {};
  int i=0;
  for (auto backend: {jordan::Backend::SparseLP,jordan::Backend::DenseJordan}) {
    jordan::Options options; options.max_iterations=200; options.tolerance=1e-6; options.backend=backend;
    auto start=std::chrono::steady_clock::now();
    const auto result=jordan::SolveSparse(lp.A,lp.b,lp.c,cones,options);
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    ASSERT_EQ(result.status,jordan::Status::Optimal) << GetParam().file << " backend=" << static_cast<int>(backend) << " it=" << result.iterations << " p=" << result.primal_residual << " d=" << result.dual_residual << " gap=" << result.gap;
    EXPECT_EQ(result.backend,backend);
    values[i++]=lp.c.dot(result.x)+lp.objective_offset;
    std::cout << std::setprecision(12) << GetParam().file << " backend=" << (backend==jordan::Backend::SparseLP?"sparse":"dense") << " rows=" << lp.A.rows() << " cols=" << lp.A.cols() << " target=" << GetParam().objective << " objective=" << values[i-1] << " primal=" << (lp.A*result.x-lp.b).norm() << " dual=" << (lp.c-lp.A.transpose()*result.y-result.s).norm() << " gap=" << result.x.dot(result.s) << " iterations=" << result.iterations << " time_s=" << seconds << '\n';
    EXPECT_NEAR(values[i-1],GetParam().objective,1e-4*std::max(1.,std::abs(GetParam().objective)));
    EXPECT_LE((lp.A*result.x-lp.b).norm(),options.tolerance*(1+lp.b.norm())*1.01);
    EXPECT_LE((lp.c-lp.A.transpose()*result.y-result.s).norm(),options.tolerance*(1+lp.c.norm())*1.01);
    EXPECT_LE(result.x.dot(result.s),options.tolerance*(1+std::abs(lp.c.dot(result.x)))*1.01);
    EXPECT_GE(result.x.minCoeff(),-1e-9); EXPECT_GE(result.s.minCoeff(),-1e-9);
  }
  EXPECT_NEAR(values[0],values[1],1e-4*std::max(1.,std::abs(GetParam().objective)));
}
INSTANTIATE_TEST_SUITE_P(Netlib,BackendComparison,::testing::Values(
    Case{"netlib/afiro.mps",-464.75314286},
    Case{"netlib/adlittle.mps",225494.96316},
    Case{"netlib/share2b.mps",-415.73224074},
    Case{"external/netlib/sc50a.mps",-64.5750770586},
    Case{"external/netlib/sc50b.mps",-70.0},
    Case{"external/netlib/blend.mps",-30.812149846}));
}  // namespace

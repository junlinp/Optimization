#include "jordan_conic.h"
#include "conic_optimality.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>


#include <cfloat>
#define JORDAN_CORE_NAMESPACE standard
#define JORDAN_CORE_REAL double
#include "jordan_core.inc"
#undef JORDAN_CORE_NAMESPACE
#undef JORDAN_CORE_REAL
#if LDBL_MANT_DIG > DBL_MANT_DIG
#define JORDAN_CORE_NAMESPACE extended
#define JORDAN_CORE_REAL long double
#include "jordan_core.inc"
#undef JORDAN_CORE_NAMESPACE
#undef JORDAN_CORE_REAL
#endif

namespace jordan {
int Dimension(const std::vector<ConeBlock>& blocks) { return detail::standard::CoreDimension(blocks); }
Eigen::VectorXd Identity(const std::vector<ConeBlock>& blocks) { return detail::standard::CoreIdentity(blocks); }
Eigen::VectorXd Product(const std::vector<ConeBlock>& blocks, const Eigen::VectorXd& x,
                        const Eigen::VectorXd& y) {
  return detail::standard::CoreProduct(blocks, x, y);
}
namespace {
template <typename T> Result Export(const T& r) {
  Result out; out.status=r.status; out.backend=r.backend; out.iterations=r.iterations;
  out.x=r.x.template cast<double>(); out.y=r.y.template cast<double>(); out.s=r.s.template cast<double>();
  out.primal_residual=r.primal_residual; out.dual_residual=r.dual_residual; out.gap=r.gap;
  return out;
}
#if LDBL_MANT_DIG > DBL_MANT_DIG
long double AccurateDot(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
  long double sum=0, correction=0;
  for (int i=0; i<x.size(); ++i) {
    const long double term=static_cast<long double>(x(i))*static_cast<long double>(y(i));
    const long double next=sum+term;
    correction += std::abs(sum)>=std::abs(term) ? (sum-next)+term : (term-next)+sum;
    sum=next;
  }
  return sum+correction;
}
// Permit only the floating-point roundoff incurred when exporting a cone
// point to double; this is independent of the requested solver tolerance.
bool ExportedConeFeasible(const std::vector<ConeBlock>& blocks, const Eigen::VectorXd& v) {
  int offset=0;
  for (const auto& block : blocks) {
    const std::vector<ConeBlock> one{block};
    const int n=Dimension(one);
    const auto slice=v.segment(offset,n);
    const long double scale=std::max(1.0, slice.cwiseAbs().maxCoeff());
    detail::extended::CoreVector point=slice.cast<long double>();
    point += (8*std::numeric_limits<double>::epsilon()*scale)*detail::extended::CoreIdentity(one);
    if (!detail::extended::Interior(one,point)) return false;
    offset+=n;
  }
  return offset==v.size();
}
Result SolveExtended(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
                     const Eigen::VectorXd& c, const std::vector<ConeBlock>& blocks,
                     const Options& options) {
  Eigen::SparseMatrix<long double> extended_A=A.cast<long double>();
  Options tighter=options; tighter.tolerance *= 0.1;
  auto core=detail::extended::SolveCore(extended_A,b.cast<long double>(),c.cast<long double>(),
                                        blocks,tighter,Backend::DenseJordan);
  Result out=Export(core);
  if (out.status!=Status::Optimal) return out;
  // Converting nearly singular PSD blocks back to double can change the sign
  // of their tiny complementarity gap. Move the dual slack minimally toward
  // the cone interior, then verify every original stopping condition again.
  long double gap=AccurateDot(out.x,out.s);
  const Eigen::VectorXd identity=Identity(blocks);
  const long double trace=AccurateDot(out.x,identity);
  if (gap<0 && trace>0) {
    const long double margin=0.1L*options.tolerance*(1+std::abs(c.dot(out.x)));
    double shift=static_cast<double>((-gap+margin)/trace);
    for (int i=0; i<4 && gap<0; ++i) {
      out.s += shift*identity;
      gap=AccurateDot(out.x,out.s);
      shift*=2;
    }
  }
  out.primal_residual=(b-A*out.x).norm();
  out.dual_residual=(c-A.transpose()*out.y-out.s).norm();
  out.gap=static_cast<double>(gap);
  if (!ExportedConeFeasible(blocks,out.x) || !ExportedConeFeasible(blocks,out.s) ||
      !ConicOptimalityReached(out.primal_residual,out.dual_residual,out.gap,
                               b.norm(),c.norm(),c.dot(out.x),options.tolerance))
    out.status=Status::NumericalFailure;
  return out;
}
#endif
bool IsOrthant(const std::vector<ConeBlock>& blocks) {
  return !blocks.empty() && std::all_of(blocks.begin(), blocks.end(),
      [](const ConeBlock& b) { return b.type == ConeType::Nonnegative; });
}
}  // namespace

Result Solve(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
             const std::vector<ConeBlock>& blocks, const Options& options) {
  if (!A.allFinite()) return {};
  return SolveSparse(A.sparseView(), b, c, blocks, options);
}

Result SolveSparse(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
                   const Eigen::VectorXd& c, const std::vector<ConeBlock>& blocks,
                   const Options& options) {
  Backend backend=options.backend;
  if (backend==Backend::Auto) backend=IsOrthant(blocks) ? Backend::SparseLP : Backend::DenseJordan;
  if (backend==Backend::SparseLP && !IsOrthant(blocks)) { Result out; out.backend=backend; return out; }
  if (backend!=Backend::SparseLP && backend!=Backend::DenseJordan) return {};
#if LDBL_MANT_DIG > DBL_MANT_DIG
  if (backend==Backend::DenseJordan && options.max_iterations>0 &&
      std::any_of(blocks.begin(),blocks.end(),[](const ConeBlock& b) {
        return b.type==ConeType::Semidefinite;
      })) return SolveExtended(A,b,c,blocks,options);
#endif
  return Export(detail::standard::SolveCore(A,b,c,blocks,options,backend));
}
}  // namespace jordan

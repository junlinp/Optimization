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

namespace jordan {
namespace {
constexpr double kSqrt2 = 1.4142135623730950488;
int BlockDim(const ConeBlock& b) {
  if (b.size <= 0 || (b.type == ConeType::Lorentz && b.size < 2)) return -1;
  if (b.type == ConeType::Semidefinite) {
    if (b.size > 46340) return -1;
    return static_cast<int>(static_cast<long long>(b.size) * (b.size + 1) / 2);
  }
  return b.type == ConeType::Nonnegative || b.type == ConeType::Lorentz ? b.size : -1;
}
Eigen::MatrixXd Mat(const Eigen::VectorXd& v, int n) {
  Eigen::MatrixXd m = Eigen::MatrixXd::Zero(n, n);
  int k = 0;
  for (int i = 0; i < n; ++i) m(i, i) = v(k++);
  for (int j = 1; j < n; ++j) for (int i = 0; i < j; ++i)
    m(i, j) = m(j, i) = v(k++) / kSqrt2;
  return m;
}
Eigen::VectorXd Vec(const Eigen::MatrixXd& m) {
  const int n = m.rows(); Eigen::VectorXd v(n * (n + 1) / 2); int k = 0;
  for (int i = 0; i < n; ++i) v(k++) = m(i, i);
  for (int j = 1; j < n; ++j) for (int i = 0; i < j; ++i)
    v(k++) = kSqrt2 * m(i, j);
  return v;
}
bool Interior(const std::vector<ConeBlock>& blocks, const Eigen::VectorXd& x) {
  int offset = 0;
  for (const auto& b : blocks) {
    int d = BlockDim(b); auto v = x.segment(offset, d);
    if (!v.allFinite()) return false;
    if (b.type == ConeType::Nonnegative) { if (v.minCoeff() <= 0) return false; }
    else if (b.type == ConeType::Lorentz) { if (v(0) <= v.tail(d - 1).norm()) return false; }
    else {
      Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> extended = Mat(v, b.size).cast<long double>();
      Eigen::SelfAdjointEigenSolver<decltype(extended)> eig(extended, Eigen::EigenvaluesOnly);
      if (eig.info() != Eigen::Success || eig.eigenvalues().minCoeff() <= 0) return false;
    }
    offset += d;
  }
  return true;
}
Eigen::MatrixXd L(const std::vector<ConeBlock>& blocks, const Eigen::VectorXd& x) {
  const int n = x.size(); Eigen::MatrixXd m(n, n);
  for (int j = 0; j < n; ++j) {
    Eigen::VectorXd e = Eigen::VectorXd::Zero(n); e(j) = 1;
    m.col(j) = Product(blocks, x, e);
  }
  return m;
}
// Spectral powers are computed in each Euclidean Jordan algebra.
Eigen::VectorXd Power(const std::vector<ConeBlock>& blocks, const Eigen::VectorXd& x, double exponent) {
  Eigen::VectorXd out(x.size()); int off = 0;
  for (const auto& b : blocks) {
    int d = BlockDim(b); Eigen::VectorXd v = x.segment(off, d);
    if (b.type == ConeType::Nonnegative) {
      if (v.minCoeff() <= 0) throw std::runtime_error("nonpositive orthant point");
      out.segment(off, d) = v.array().pow(exponent);
    } else if (b.type == ConeType::Lorentz) {
      double r = v.tail(d - 1).norm(), a = v(0) + r, z = v(0) - r;
      if (z <= 0) throw std::runtime_error("noninterior Lorentz point");
      double ap = std::pow(a, exponent), zp = std::pow(z, exponent);
      out(off) = (ap + zp) / 2;
      if (r > 0) out.segment(off + 1, d - 1) = ((ap - zp) / (2*r)) * v.tail(d - 1);
      else out.segment(off + 1, d - 1).setZero();
    } else {
      Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> extended = Mat(v, b.size).cast<long double>();
      Eigen::SelfAdjointEigenSolver<decltype(extended)> eig(extended);
      if (eig.info() != Eigen::Success) throw std::runtime_error("PSD eigendecomposition failed");
      const long double largest = eig.eigenvalues().maxCoeff();
      const long double floor = std::numeric_limits<double>::epsilon() * std::max(1.0L, largest);
      if (largest <= 0 || eig.eigenvalues().minCoeff() < -floor)
        throw std::runtime_error("noninterior PSD point");
      // Roundoff-level spectral regularization affects only the scaling map;
      // the actual iterates and the original KKT residuals are not modified.
      auto regularized = eig.eigenvalues().cwiseMax(floor);
      auto result = (eig.eigenvectors() * regularized.array().pow(static_cast<long double>(exponent)).matrix().asDiagonal() * eig.eigenvectors().transpose()).eval();
      out.segment(off, d) = Vec(result.cast<double>());
    }
    off += d;
  }
  return out;
}
Eigen::VectorXd ApplyQuadratic(const std::vector<ConeBlock>& blocks,
                               const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
  Eigen::VectorXd out(x.size()); int off = 0;
  for (const auto& b : blocks) {
    int d = BlockDim(b); Eigen::VectorXd a = x.segment(off, d), v = y.segment(off, d);
    if (b.type == ConeType::Nonnegative) out.segment(off, d) = a.array().square() * v.array();
    else if (b.type == ConeType::Lorentz) {
      const double t = a(0), r2 = a.tail(d-1).squaredNorm();
      const double dot = a.tail(d-1).dot(v.tail(d-1));
      out(off) = (t*t+r2)*v(0) + 2*t*dot;
      out.segment(off+1,d-1) = 2*t*v(0)*a.tail(d-1) + (t*t-r2)*v.tail(d-1) + 2*dot*a.tail(d-1);
    } else {
      Eigen::MatrixXd X = Mat(a, b.size), Y = Mat(v, b.size);
      Eigen::MatrixXd product = X * Y * X;
      out.segment(off, d) = Vec((product + product.transpose()).eval() * 0.5);
    }
    off += d;
  }
  return out;
}
Eigen::MatrixXd Quadratic(const std::vector<ConeBlock>& blocks, const Eigen::VectorXd& x) {
  Eigen::MatrixXd out(x.size(), x.size());
  for (int j = 0; j < x.size(); ++j) {
    Eigen::VectorXd e = Eigen::VectorXd::Zero(x.size()); e(j) = 1;
    out.col(j) = ApplyQuadratic(blocks, x, e);
  }
  return (out + out.transpose()).eval() * 0.5;
}
// Repair only roundoff-level loss of PSD interior. The adjusted iterates
// are used to recompute original-problem KKT residuals before convergence.
bool RepairRoundoffPSD(const std::vector<ConeBlock>& blocks, Eigen::VectorXd* x) {
  int off=0;
  for (const auto& block: blocks) {
    const int d=BlockDim(block);
    if (block.type == ConeType::Semidefinite) {
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(Mat(x->segment(off,d),block.size),Eigen::EigenvaluesOnly);
      if (eig.info()!=Eigen::Success) return false;
      const double scale=std::max(1.0,eig.eigenvalues().cwiseAbs().maxCoeff());
      const double min=eig.eigenvalues().minCoeff();
      if (min <= 0) {
        const double eps=std::numeric_limits<double>::epsilon()*scale;
        if (min < -8*eps) return false;
        x->segment(off,block.size).array() += -min + eps;
      }
    }
    off+=d;
  }
  return true;
}
// Compute PSD NT scaling through SVD factors rather than forming ill-conditioned
// positive-definite Gram matrices and subtracting tiny eigenvalues in roundoff.
void NTScaling(const std::vector<ConeBlock>& blocks, const Eigen::VectorXd& x,
               const Eigen::VectorXd& s, Eigen::MatrixXd* W, Eigen::MatrixXd* Wi) {
  W->setZero(x.size(), x.size()); Wi->setZero(x.size(), x.size());
  int off=0;
  for (const auto& block : blocks) {
    int d=BlockDim(block); std::vector<ConeBlock> single{block};
    Eigen::VectorXd xb=x.segment(off,d), sb=s.segment(off,d);
    Eigen::VectorXd sqrt_w, invsqrt_w;
    if (block.type == ConeType::Semidefinite) {
      Eigen::MatrixXd Xh=Mat(Power(single,xb,0.5),block.size);
      Eigen::MatrixXd Sh=Mat(Power(single,sb,0.5),block.size);
      Eigen::JacobiSVD<Eigen::MatrixXd> first(Xh*Sh,Eigen::ComputeFullU);
      if (first.info()!=Eigen::Success || first.singularValues().minCoeff()<=0)
        throw std::runtime_error("singular PSD scaling factor");
      Eigen::MatrixXd invquarter=first.matrixU()*first.singularValues().array().sqrt().inverse().matrix().asDiagonal()*first.matrixU().transpose();
      Eigen::JacobiSVD<Eigen::MatrixXd> second(Xh*invquarter,Eigen::ComputeFullU);
      if (second.info()!=Eigen::Success || second.singularValues().minCoeff()<=0)
        throw std::runtime_error("singular PSD scaling point");
      Eigen::MatrixXd wh=second.matrixU()*second.singularValues().asDiagonal()*second.matrixU().transpose();
      Eigen::MatrixXd wih=second.matrixU()*second.singularValues().cwiseInverse().asDiagonal()*second.matrixU().transpose();
      sqrt_w=Vec((wh+wh.transpose()).eval()*0.5);
      invsqrt_w=Vec((wih+wih.transpose()).eval()*0.5);
    } else {
      Eigen::VectorXd xh=Power(single,xb,0.5);
      Eigen::VectorXd v=ApplyQuadratic(single,xh,sb);
      Eigen::VectorXd w=ApplyQuadratic(single,xh,Power(single,v,-0.5));
      sqrt_w=Power(single,w,0.5);
      invsqrt_w=Power(single,w,-0.5);
    }
    W->block(off,off,d,d)=Quadratic(single,sqrt_w);
    Wi->block(off,off,d,d)=Quadratic(single,invsqrt_w);
    off+=d;
  }
}
// Equilibrate the Newton equations without altering their solution.
class EquilibratedLU {
 public:
  explicit EquilibratedLU(const Eigen::MatrixXd& A) : original_(A), scaled_(A),
      rows_(Eigen::VectorXd::Ones(A.rows())), cols_(Eigen::VectorXd::Ones(A.cols())) {
    for (int k = 0; k < 6; ++k) {
      for (int i = 0; i < scaled_.rows(); ++i) {
        double v = scaled_.row(i).cwiseAbs().maxCoeff();
        if (v > 0) { double f = 1/std::sqrt(v); scaled_.row(i) *= f; rows_(i) *= f; }
      }
      for (int j = 0; j < scaled_.cols(); ++j) {
        double v = scaled_.col(j).cwiseAbs().maxCoeff();
        if (v > 0) { double f = 1/std::sqrt(v); scaled_.col(j) *= f; cols_(j) *= f; }
      }
    }
    lu_.setThreshold(1e-20);
    lu_.compute(scaled_);
  }
  bool invertible() const { return lu_.isInvertible(); }
  Eigen::VectorXd solve(const Eigen::VectorXd& rhs) const {
    Eigen::VectorXd x = cols_.cwiseProduct(lu_.solve(rows_.cwiseProduct(rhs)));
    for (int k=0; k<2; ++k) {
      Eigen::VectorXd residual = rhs - original_ * x;
      if (!residual.allFinite()) break;
      x += cols_.cwiseProduct(lu_.solve(rows_.cwiseProduct(residual)));
    }
    return x;
  }
 private:
  Eigen::MatrixXd original_, scaled_;
  Eigen::VectorXd rows_, cols_;
  Eigen::FullPivLU<Eigen::MatrixXd> lu_;
};
double Step(const std::vector<ConeBlock>& blocks, const Eigen::VectorXd& x, const Eigen::VectorXd& dx) {
  if (Interior(blocks, x + dx)) return 1;
  double lo = 0, hi = 1;
  for (int i = 0; i < 45; ++i) {
    double mid = (lo + hi)/2;
    if (Interior(blocks, x + mid * dx)) lo = mid; else hi = mid;
  }
  return lo;
}
}  // namespace
int Dimension(const std::vector<ConeBlock>& blocks) {
  long long n = 0;
  if (blocks.empty()) return -1;
  for (const auto& b : blocks) { int d = BlockDim(b); if (d < 0) return -1; n += d; }
  return n > std::numeric_limits<int>::max() ? -1 : static_cast<int>(n);
}
Eigen::VectorXd Identity(const std::vector<ConeBlock>& blocks) {
  int n = Dimension(blocks); if (n < 0) throw std::invalid_argument("invalid cone blocks");
  Eigen::VectorXd e = Eigen::VectorXd::Zero(n); int off = 0;
  for (const auto& b : blocks) {
    if (b.type == ConeType::Lorentz) e(off) = 1;
    else if (b.type == ConeType::Nonnegative) e.segment(off, b.size).setOnes();
    else e.segment(off, b.size).setOnes();
    off += BlockDim(b);
  }
  return e;
}
Eigen::VectorXd Product(const std::vector<ConeBlock>& blocks, const Eigen::VectorXd& x,
                        const Eigen::VectorXd& y) {
  const int n = Dimension(blocks);
  if (n < 0 || x.size() != n || y.size() != n) throw std::invalid_argument("invalid Jordan product dimensions");
  Eigen::VectorXd z(n); int off = 0;
  for (const auto& b : blocks) {
    int d = BlockDim(b); auto a = x.segment(off, d); auto v = y.segment(off, d);
    if (b.type == ConeType::Nonnegative) z.segment(off, d) = a.cwiseProduct(v);
    else if (b.type == ConeType::Lorentz) {
      z(off) = a.dot(v);
      z.segment(off + 1, d - 1) = a(0) * v.tail(d - 1) + v(0) * a.tail(d - 1);
    } else {
      Eigen::MatrixXd A = Mat(a, b.size), B = Mat(v, b.size);
      z.segment(off, d) = Vec((A * B + B * A) * 0.5);
    }
    off += d;
  }
  return z;
}
static Result SolveCore(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
             const std::vector<ConeBlock>& blocks, const Options& options, Backend backend) {
  Result out; out.backend = backend; const int n = Dimension(blocks), m = A.rows();
  const bool sparse = backend == Backend::SparseLP;
  if (n <= 0 || A.cols() != n || b.size() != m || c.size() != n ||
      options.max_iterations < 0 || !std::isfinite(options.tolerance) || options.tolerance <= 0 ||
      !A.coeffs().allFinite() || !b.allFinite() || !c.allFinite()) return out;
  // Equilibrate rows and cone-compatible columns. SOC blocks use scalar
  // scaling; PSD blocks use diagonal congruences; orthant coordinates scale
  // independently. x = D_c x_s, y = D_r y_s, s = D_c^-1 s_s.
  Eigen::MatrixXd As;
  Eigen::SparseMatrix<double> Asp;
  if (sparse) Asp = A; else As = Eigen::MatrixXd(A);
  Eigen::VectorXd bs = b, cs = c;
  Eigen::VectorXd Dr = Eigen::VectorXd::Ones(m), Dc = Eigen::VectorXd::Ones(n);
  for (int pass=0; pass<8; ++pass) {
    if (sparse) {
      Eigen::VectorXd rowmag = bs.cwiseAbs();
      for (int k=0;k<Asp.outerSize();++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(Asp,k);it;++it)
          rowmag(it.row())=std::max(rowmag(it.row()),std::abs(it.value()));
      Eigen::VectorXd rf = rowmag.array().max(1e-12).sqrt().inverse().max(1e-3).min(1e3);
      bs.array() *= rf.array(); Dr.array() *= rf.array();
      for (int k=0;k<Asp.outerSize();++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(Asp,k);it;++it) it.valueRef()*=rf(it.row());
      Eigen::VectorXd colmag = cs.cwiseAbs();
      for (int k=0;k<Asp.outerSize();++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(Asp,k);it;++it)
          colmag(it.col())=std::max(colmag(it.col()),std::abs(it.value()));
      Eigen::VectorXd cf = colmag.array().max(1e-12).sqrt().inverse().max(1e-3).min(1e3);
      cs.array() *= cf.array(); Dc.array() *= cf.array();
      for (int k=0;k<Asp.outerSize();++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(Asp,k);it;++it) it.valueRef()*=cf(it.col());
      continue;
    }
    for (int i=0; i<m; ++i) {
      double mag = std::max(As.row(i).cwiseAbs().maxCoeff(), std::abs(bs(i)));
      double f = 1/std::sqrt(std::max(mag, 1e-12));
      f = std::max(1e-3, std::min(1e3, f));
      As.row(i) *= f; bs(i) *= f; Dr(i) *= f;
    }
    int off=0;
    for (const auto& block : blocks) {
      int d=BlockDim(block);
      if (block.type == ConeType::Semidefinite) {
        Eigen::VectorXd magnitude = cs.segment(off,d).cwiseAbs();
        if (m > 0) for (int j=0;j<d;++j)
          magnitude(j) = std::max(magnitude(j), As.col(off+j).cwiseAbs().maxCoeff());
        Eigen::VectorXd vertex = magnitude.head(block.size);
        int k=block.size;
        for (int col=1;col<block.size;++col) for (int row=0;row<col;++row) {
          vertex(row)=std::max(vertex(row),magnitude(k));
          vertex(col)=std::max(vertex(col),magnitude(k)); ++k;
        }
        Eigen::VectorXd factors(block.size);
        for (int j=0;j<block.size;++j) factors(j)=std::max(1e-3,std::min(1e3,std::pow(std::max(vertex(j),1e-12),-0.25)));
        Eigen::VectorXd scale(d); scale.head(block.size)=factors.array().square().matrix();
        k=block.size;
        for (int col=1;col<block.size;++col) for (int row=0;row<col;++row) scale(k++)=factors(row)*factors(col);
        As.middleCols(off,d) *= scale.asDiagonal(); cs.segment(off,d).array() *= scale.array(); Dc.segment(off,d).array() *= scale.array();
      } else {
        for (int j=0; j<d;) {
          int width = block.type == ConeType::Nonnegative ? 1 : d;
          double mag = cs.segment(off+j,width).cwiseAbs().maxCoeff();
          if (m > 0) mag = std::max(mag, As.middleCols(off+j,width).cwiseAbs().maxCoeff());
          double f = 1/std::sqrt(std::max(mag, 1e-12));
          f = std::max(1e-3, std::min(1e3, f));
          As.middleCols(off+j,width) *= f; cs.segment(off+j,width) *= f; Dc.segment(off+j,width) *= f;
          j += width;
        }
      }
      off += d;
    }
  }
  const Eigen::VectorXd e = Identity(blocks);
  const bool orthant = std::all_of(blocks.begin(), blocks.end(),
      [](const ConeBlock& block) { return block.type == ConeType::Nonnegative; });
  out.x = orthant ? 5.0 * e : e; out.s = out.x; out.y = Eigen::VectorXd::Zero(m);
  out.status = Status::MaxIterations;
  for (int iter = 0; iter <= options.max_iterations; ++iter) {
    out.iterations = iter;
    if (!RepairRoundoffPSD(blocks, &out.x) || !RepairRoundoffPSD(blocks, &out.s)) {
      out.status=Status::NumericalFailure; break;
    }
    Eigen::VectorXd rp = bs - (sparse ? Eigen::VectorXd(Asp*out.x) : Eigen::VectorXd(As*out.x));
    Eigen::VectorXd rd = cs - (sparse ? Eigen::VectorXd(Asp.transpose()*out.y) : Eigen::VectorXd(As.transpose()*out.y)) - out.s;
    const Eigen::VectorXd original_x = Dc.cwiseProduct(out.x);
    const Eigen::VectorXd original_y = Dr.cwiseProduct(out.y);
    const Eigen::VectorXd original_s = out.s.cwiseQuotient(Dc);
    out.primal_residual = (b - A*original_x).norm();
    out.dual_residual = (c - A.transpose()*original_y - original_s).norm();
    out.gap = original_x.dot(original_s);
    if (!std::isfinite(out.gap) || !rp.allFinite() || !rd.allFinite()) { out.status = Status::NumericalFailure; break; }
    if (ConicOptimalityReached(out.primal_residual, out.dual_residual, out.gap,
                               b.norm(), c.norm(), c.dot(original_x), options.tolerance)) {
      out.status = Status::Optimal; break;
    }
    if (iter == options.max_iterations) break;
    // NT scaling W maps the primal and dual variables to the same point:
    // W^-1 x = W s = lambda. P(z)=2L(z)^2-L(z o z).
    try {
      Eigen::MatrixXd W, Wi;
      Eigen::VectorXd w, wi, lambda;
      if (sparse) {
        // Orthant specialization of the same Jordan quadratic scaling:
        // W=P(sqrt(w))=diag(sqrt(x/s)), lambda=W*s.
        w=(out.x.array()/out.s.array()).sqrt(); wi=w.cwiseInverse();
        lambda=w.cwiseProduct(out.s);
      } else {
        NTScaling(blocks, out.x, out.s, &W, &Wi);
        lambda = W * out.s;
      }
      Eigen::VectorXd l2 = Product(blocks, lambda, lambda);
      Eigen::FullPivLU<Eigen::MatrixXd> lambda_lu;
      if (!sparse) {
        lambda_lu.compute(L(blocks, lambda));
        if (!lambda_lu.isInvertible()) { out.status = Status::NumericalFailure; break; }
      }
      // Both paths solve A W u=rp, A^T dy+W^-1 v=rd, u+v=q.
      Eigen::MatrixXd K;
      std::unique_ptr<EquilibratedLU> dense_lu;
      Eigen::SparseMatrix<double> normal;
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> sparse_ldlt;
      Eigen::SparseLU<Eigen::SparseMatrix<double>> sparse_lu;
      bool use_ldlt=false;
      if (sparse && m > 0) {
        Eigen::SparseMatrix<double> scaled=Asp;
        for (int k=0;k<scaled.outerSize();++k)
          for (Eigen::SparseMatrix<double>::InnerIterator it(scaled,k);it;++it)
            it.valueRef()*=w(it.col())*w(it.col());
        normal=scaled*Asp.transpose(); normal.makeCompressed();
        sparse_ldlt.compute(normal); use_ldlt=sparse_ldlt.info()==Eigen::Success;
        if (!use_ldlt) {
          // Regularization handles dependent equality rows. The actual Newton
          // residual is still checked below against the unregularized system.
          for (int i=0;i<m;++i) normal.coeffRef(i,i) += 1e-10;
          normal.makeCompressed(); sparse_lu.compute(normal);
          if (sparse_lu.info()!=Eigen::Success) { out.status=Status::NumericalFailure; break; }
        }
      } else if (!sparse) {
        K = Eigen::MatrixXd::Zero(m+2*n,m+2*n);
        K.block(0,0,m,n)=As*W; K.block(m,n,n,m)=As.transpose(); K.block(m,n+m,n,n)=Wi;
        K.block(m+n,0,n,n).setIdentity(); K.block(m+n,n+m,n,n).setIdentity();
        dense_lu=std::make_unique<EquilibratedLU>(K);
        if (!dense_lu->invertible()) { out.status=Status::NumericalFailure; break; }
      }
      auto solve_direction = [&](const Eigen::VectorXd& rc, Eigen::VectorXd* u,
                                 Eigen::VectorXd* dy, Eigen::VectorXd* v) {
        Eigen::VectorXd q = sparse ? rc.cwiseQuotient(lambda) : lambda_lu.solve(rc).eval();
        if (sparse) {
          Eigen::VectorXd nrhs = rp - Asp*w.cwiseProduct(q) + Asp*(w.array().square().matrix().cwiseProduct(rd));
          if (m==0) dy->resize(0);
          else if (use_ldlt) *dy=sparse_ldlt.solve(nrhs);
          else *dy=sparse_lu.solve(nrhs);
          *v=w.cwiseProduct(rd-Asp.transpose()*(*dy)); *u=q-*v;
          const double direction_error = (Asp*w.cwiseProduct(*u)-rp).norm();
          if (std::getenv("JORDAN_TRACE")) std::cerr << "sparse direction residual=" << direction_error/(1+rp.norm()) << "\n";
          return u->allFinite() && dy->allFinite() && v->allFinite() &&
              direction_error <= 1e-5*(1+rp.norm());
        }
        Eigen::VectorXd rhs(m+2*n); rhs << rp,rd,q;
        Eigen::VectorXd direction=dense_lu->solve(rhs);
        if (!direction.allFinite() || (K*direction-rhs).norm()>1e-7*(1+rhs.norm())) return false;
        *u=direction.head(n); *dy=direction.segment(n,m); *v=direction.tail(n); return true;
      };
      Eigen::VectorXd au, ay, av;
      if (!solve_direction(-l2,&au,&ay,&av)) { out.status=Status::NumericalFailure; break; }
      Eigen::VectorXd dx_aff = sparse ? w.cwiseProduct(au) : Eigen::VectorXd(W*au);
      Eigen::VectorXd ds_aff = sparse ? wi.cwiseProduct(av) : Eigen::VectorXd(Wi*av);
      double ap_aff = Step(blocks, out.x, dx_aff), ad_aff = Step(blocks, out.s, ds_aff);
      double mu = out.x.dot(out.s) / e.squaredNorm();
      double mu_aff = (out.x + ap_aff*dx_aff).dot(out.s + ad_aff*ds_aff) / e.squaredNorm();
      double sigma = std::pow(std::max(0.0, std::min(1.0, mu_aff/mu)), 3);
      if (!orthant) sigma = std::max(0.1, sigma);
      Eigen::VectorXd correction = Product(blocks, au, av);
      Eigen::VectorXd u, dy, v;
      if (!solve_direction(sigma*mu*e-l2-correction,&u,&dy,&v)) {
        out.status=Status::NumericalFailure; break;
      }
      Eigen::VectorXd dx = sparse ? w.cwiseProduct(u) : Eigen::VectorXd(W*u);
      Eigen::VectorXd ds = sparse ? wi.cwiseProduct(v) : Eigen::VectorXd(Wi*v);
      const double fraction = orthant ? 0.995 : 0.9;
      double ap = std::min(1.0, fraction * Step(blocks, out.x, dx));
      double ad = std::min(1.0, fraction * Step(blocks, out.s, ds));
      if (std::getenv("JORDAN_TRACE")) std::cerr << "iter=" << iter << " rp=" << out.primal_residual << " rd=" << out.dual_residual << " gap=" << out.gap << " ap=" << ap << " ad=" << ad << " sigma=" << sigma << "\n";
      if (ap <= 1e-14 || ad <= 1e-14) { out.status = Status::NumericalFailure; break; }
      out.x += ap * dx; out.y += ad * dy; out.s += ad * ds;

    } catch (const std::exception& error) { if (std::getenv("JORDAN_TRACE")) std::cerr << error.what() << "\n"; out.status = Status::NumericalFailure; break; }
  }
  out.x = Dc.cwiseProduct(out.x);
  out.y = Dr.cwiseProduct(out.y);
  out.s = out.s.cwiseQuotient(Dc);
  // Include any roundoff repair, including a repair preceding failure.
  out.primal_residual = (b - A*out.x).norm();
  out.dual_residual = (c - A.transpose()*out.y - out.s).norm();
  out.gap = out.x.dot(out.s);
  if (out.status == Status::Optimal &&
      !ConicOptimalityReached(out.primal_residual, out.dual_residual, out.gap,
                              b.norm(), c.norm(), c.dot(out.x), options.tolerance))
    out.status = Status::NumericalFailure;
  return out;
}
namespace {
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
  return SolveCore(A,b,c,blocks,options,backend);
}
}  // namespace jordan

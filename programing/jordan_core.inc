// Included twice with scalar and namespace macros. One implementation keeps
// the ordinary and extended-precision predictor-corrector equations identical.
namespace jordan { namespace detail { namespace JORDAN_CORE_NAMESPACE {
using Real = JORDAN_CORE_REAL;
using CoreVector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using CoreMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
using CoreSparse = Eigen::SparseMatrix<Real>;
struct CoreResult { Status status=Status::InvalidInput; Backend backend=Backend::Auto; CoreVector x,y,s; int iterations=0; Real primal_residual=0, dual_residual=0, gap=0; };
CoreVector CoreProduct(const std::vector<ConeBlock>&,const CoreVector&,const CoreVector&);

namespace {
constexpr Real kSqrt2 = Real(1.4142135623730950488);
int BlockDim(const ConeBlock& b) {
  if (b.size <= 0 || (b.type == ConeType::Lorentz && b.size < 2)) return -1;
  if (b.type == ConeType::Semidefinite) {
    if (b.size > 46340) return -1;
    return static_cast<int>(static_cast<long long>(b.size) * (b.size + 1) / 2);
  }
  return b.type == ConeType::Nonnegative || b.type == ConeType::Lorentz ? b.size : -1;
}
CoreMatrix Mat(const CoreVector& v, int n) {
  CoreMatrix m = CoreMatrix::Zero(n, n);
  int k = 0;
  for (int i = 0; i < n; ++i) m(i, i) = v(k++);
  for (int j = 1; j < n; ++j) for (int i = 0; i < j; ++i)
    m(i, j) = m(j, i) = v(k++) / kSqrt2;
  return m;
}
CoreVector Vec(const CoreMatrix& m) {
  const int n = m.rows(); CoreVector v(n * (n + 1) / 2); int k = 0;
  for (int i = 0; i < n; ++i) v(k++) = m(i, i);
  for (int j = 1; j < n; ++j) for (int i = 0; i < j; ++i)
    v(k++) = kSqrt2 * m(i, j);
  return v;
}
bool Interior(const std::vector<ConeBlock>& blocks, const CoreVector& x) {
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
CoreMatrix L(const std::vector<ConeBlock>& blocks, const CoreVector& x) {
  const int n = x.size(); CoreMatrix m(n, n);
  for (int j = 0; j < n; ++j) {
    CoreVector e = CoreVector::Zero(n); e(j) = 1;
    m.col(j) = CoreProduct(blocks, x, e);
  }
  return m;
}
// Spectral powers are computed in each Euclidean Jordan algebra.
CoreVector Power(const std::vector<ConeBlock>& blocks, const CoreVector& x, Real exponent) {
  CoreVector out(x.size()); int off = 0;
  for (const auto& b : blocks) {
    int d = BlockDim(b); CoreVector v = x.segment(off, d);
    if (b.type == ConeType::Nonnegative) {
      if (v.minCoeff() <= 0) throw std::runtime_error("nonpositive orthant point");
      out.segment(off, d) = v.array().pow(exponent);
    } else if (b.type == ConeType::Lorentz) {
      Real r = v.tail(d - 1).norm(), a = v(0) + r, z = v(0) - r;
      if (z <= 0) throw std::runtime_error("noninterior Lorentz point");
      Real ap = std::pow(a, exponent), zp = std::pow(z, exponent);
      out(off) = (ap + zp) / 2;
      if (r > 0) out.segment(off + 1, d - 1) = ((ap - zp) / (2*r)) * v.tail(d - 1);
      else out.segment(off + 1, d - 1).setZero();
    } else {
      Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> extended = Mat(v, b.size).cast<long double>();
      Eigen::SelfAdjointEigenSolver<decltype(extended)> eig(extended);
      if (eig.info() != Eigen::Success) throw std::runtime_error("PSD eigendecomposition failed");
      const long double largest = eig.eigenvalues().maxCoeff();
      const long double floor = std::numeric_limits<Real>::epsilon() * std::max(1.0L, largest);
      if (largest <= 0 || eig.eigenvalues().minCoeff() < -floor)
        throw std::runtime_error("noninterior PSD point");
      // Roundoff-level spectral regularization affects only the scaling map;
      // the actual iterates and the original KKT residuals are not modified.
      auto regularized = eig.eigenvalues().cwiseMax(floor);
      auto result = (eig.eigenvectors() * regularized.array().pow(static_cast<long double>(exponent)).matrix().asDiagonal() * eig.eigenvectors().transpose()).eval();
      out.segment(off, d) = Vec(result.cast<Real>());
    }
    off += d;
  }
  return out;
}
CoreVector ApplyQuadratic(const std::vector<ConeBlock>& blocks,
                               const CoreVector& x, const CoreVector& y) {
  CoreVector out(x.size()); int off = 0;
  for (const auto& b : blocks) {
    int d = BlockDim(b); CoreVector a = x.segment(off, d), v = y.segment(off, d);
    if (b.type == ConeType::Nonnegative) out.segment(off, d) = a.array().square() * v.array();
    else if (b.type == ConeType::Lorentz) {
      const Real t = a(0), r2 = a.tail(d-1).squaredNorm();
      const Real dot = a.tail(d-1).dot(v.tail(d-1));
      out(off) = (t*t+r2)*v(0) + 2*t*dot;
      out.segment(off+1,d-1) = 2*t*v(0)*a.tail(d-1) + (t*t-r2)*v.tail(d-1) + 2*dot*a.tail(d-1);
    } else {
      CoreMatrix X = Mat(a, b.size), Y = Mat(v, b.size);
      CoreMatrix product = X * Y * X;
      out.segment(off, d) = Vec((product + product.transpose()).eval() * Real(0.5));
    }
    off += d;
  }
  return out;
}
CoreMatrix Quadratic(const std::vector<ConeBlock>& blocks, const CoreVector& x) {
  CoreMatrix out(x.size(), x.size());
  for (int j = 0; j < x.size(); ++j) {
    CoreVector e = CoreVector::Zero(x.size()); e(j) = 1;
    out.col(j) = ApplyQuadratic(blocks, x, e);
  }
  return (out + out.transpose()).eval() * Real(0.5);
}
// Repair only roundoff-level loss of PSD interior. The adjusted iterates
// are used to recompute original-problem KKT residuals before convergence.
bool RepairRoundoffPSD(const std::vector<ConeBlock>& blocks, CoreVector* x) {
  int off=0;
  for (const auto& block: blocks) {
    const int d=BlockDim(block);
    if (block.type == ConeType::Semidefinite) {
      Eigen::SelfAdjointEigenSolver<CoreMatrix> eig(Mat(x->segment(off,d),block.size),Eigen::EigenvaluesOnly);
      if (eig.info()!=Eigen::Success) return false;
      const Real scale=std::max(Real(1.0),eig.eigenvalues().cwiseAbs().maxCoeff());
      const Real min=eig.eigenvalues().minCoeff();
      if (min <= 0) {
        const Real eps=std::numeric_limits<Real>::epsilon()*scale;
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
void NTScaling(const std::vector<ConeBlock>& blocks, const CoreVector& x,
               const CoreVector& s, CoreMatrix* W, CoreMatrix* Wi) {
  W->setZero(x.size(), x.size()); Wi->setZero(x.size(), x.size());
  int off=0;
  for (const auto& block : blocks) {
    int d=BlockDim(block); std::vector<ConeBlock> single{block};
    CoreVector xb=x.segment(off,d), sb=s.segment(off,d);
    CoreVector sqrt_w, invsqrt_w;
    if (block.type == ConeType::Semidefinite) {
      CoreMatrix Xh=Mat(Power(single,xb,Real(0.5)),block.size);
      CoreMatrix Sh=Mat(Power(single,sb,Real(0.5)),block.size);
      Eigen::JacobiSVD<CoreMatrix> first(Xh*Sh,Eigen::ComputeFullU);
      if (first.info()!=Eigen::Success || first.singularValues().minCoeff()<=0)
        throw std::runtime_error("singular PSD scaling factor");
      CoreMatrix invquarter=first.matrixU()*first.singularValues().array().sqrt().inverse().matrix().asDiagonal()*first.matrixU().transpose();
      Eigen::JacobiSVD<CoreMatrix> second(Xh*invquarter,Eigen::ComputeFullU);
      if (second.info()!=Eigen::Success || second.singularValues().minCoeff()<=0)
        throw std::runtime_error("singular PSD scaling point");
      CoreMatrix wh=second.matrixU()*second.singularValues().asDiagonal()*second.matrixU().transpose();
      CoreMatrix wih=second.matrixU()*second.singularValues().cwiseInverse().asDiagonal()*second.matrixU().transpose();
      sqrt_w=Vec((wh+wh.transpose()).eval()*Real(0.5));
      invsqrt_w=Vec((wih+wih.transpose()).eval()*Real(0.5));
    } else {
      CoreVector xh=Power(single,xb,Real(0.5));
      CoreVector v=ApplyQuadratic(single,xh,sb);
      CoreVector w=ApplyQuadratic(single,xh,Power(single,v,-Real(0.5)));
      sqrt_w=Power(single,w,Real(0.5));
      invsqrt_w=Power(single,w,-Real(0.5));
    }
    W->block(off,off,d,d)=Quadratic(single,sqrt_w);
    Wi->block(off,off,d,d)=Quadratic(single,invsqrt_w);
    off+=d;
  }
}
// Equilibrate the Newton equations without altering their solution.
class EquilibratedLU {
 public:
  explicit EquilibratedLU(const CoreMatrix& A) : original_(A), scaled_(A),
      rows_(CoreVector::Ones(A.rows())), cols_(CoreVector::Ones(A.cols())) {
    for (int k = 0; k < 6; ++k) {
      for (int i = 0; i < scaled_.rows(); ++i) {
        Real v = scaled_.row(i).cwiseAbs().maxCoeff();
        if (v > 0) { Real f = 1/std::sqrt(v); scaled_.row(i) *= f; rows_(i) *= f; }
      }
      for (int j = 0; j < scaled_.cols(); ++j) {
        Real v = scaled_.col(j).cwiseAbs().maxCoeff();
        if (v > 0) { Real f = 1/std::sqrt(v); scaled_.col(j) *= f; cols_(j) *= f; }
      }
    }
    lu_.setThreshold(Real(1e-20));
    lu_.compute(scaled_);
  }
  bool invertible() const { return lu_.isInvertible(); }
  CoreVector solve(const CoreVector& rhs) const {
    CoreVector x = cols_.cwiseProduct(lu_.solve(rows_.cwiseProduct(rhs)));
    for (int k=0; k<2; ++k) {
      CoreVector residual = rhs - original_ * x;
      if (!residual.allFinite()) break;
      x += cols_.cwiseProduct(lu_.solve(rows_.cwiseProduct(residual)));
    }
    return x;
  }
 private:
  CoreMatrix original_, scaled_;
  CoreVector rows_, cols_;
  Eigen::FullPivLU<CoreMatrix> lu_;
};
Real Step(const std::vector<ConeBlock>& blocks, const CoreVector& x, const CoreVector& dx) {
  if (Interior(blocks, x + dx)) return 1;
  Real lo = 0, hi = 1;
  for (int i = 0; i < 45; ++i) {
    Real mid = (lo + hi)/2;
    if (Interior(blocks, x + mid * dx)) lo = mid; else hi = mid;
  }
  return lo;
}
}  // namespace
int CoreDimension(const std::vector<ConeBlock>& blocks) {
  long long n = 0;
  if (blocks.empty()) return -1;
  for (const auto& b : blocks) { int d = BlockDim(b); if (d < 0) return -1; n += d; }
  return n > std::numeric_limits<int>::max() ? -1 : static_cast<int>(n);
}
CoreVector CoreIdentity(const std::vector<ConeBlock>& blocks) {
  int n = CoreDimension(blocks); if (n < 0) throw std::invalid_argument("invalid cone blocks");
  CoreVector e = CoreVector::Zero(n); int off = 0;
  for (const auto& b : blocks) {
    if (b.type == ConeType::Lorentz) e(off) = 1;
    else if (b.type == ConeType::Nonnegative) e.segment(off, b.size).setOnes();
    else e.segment(off, b.size).setOnes();
    off += BlockDim(b);
  }
  return e;
}
CoreVector CoreProduct(const std::vector<ConeBlock>& blocks, const CoreVector& x,
                        const CoreVector& y) {
  const int n = CoreDimension(blocks);
  if (n < 0 || x.size() != n || y.size() != n) throw std::invalid_argument("invalid Jordan product dimensions");
  CoreVector z(n); int off = 0;
  for (const auto& b : blocks) {
    int d = BlockDim(b); auto a = x.segment(off, d); auto v = y.segment(off, d);
    if (b.type == ConeType::Nonnegative) z.segment(off, d) = a.cwiseProduct(v);
    else if (b.type == ConeType::Lorentz) {
      z(off) = a.dot(v);
      z.segment(off + 1, d - 1) = a(0) * v.tail(d - 1) + v(0) * a.tail(d - 1);
    } else {
      CoreMatrix A = Mat(a, b.size), B = Mat(v, b.size);
      z.segment(off, d) = Vec((A * B + B * A) * Real(0.5));
    }
    off += d;
  }
  return z;
}
static CoreResult SolveCore(const CoreSparse& A, const CoreVector& b, const CoreVector& c,
             const std::vector<ConeBlock>& blocks, const Options& options, Backend backend) {
  CoreResult out; out.backend = backend; const int n = CoreDimension(blocks), m = A.rows();
  const bool sparse = backend == Backend::SparseLP;
  if (n <= 0 || A.cols() != n || b.size() != m || c.size() != n ||
      options.max_iterations < 0 || !std::isfinite(options.tolerance) || options.tolerance <= 0 ||
      !A.coeffs().allFinite() || !b.allFinite() || !c.allFinite()) return out;
  // Equilibrate rows and cone-compatible columns. SOC blocks use scalar
  // scaling; PSD blocks use diagonal congruences; orthant coordinates scale
  // independently. x = D_c x_s, y = D_r y_s, s = D_c^-1 s_s.
  CoreMatrix As;
  CoreSparse Asp;
  if (sparse) Asp = A; else As = CoreMatrix(A);
  CoreVector bs = b, cs = c;
  CoreVector Dr = CoreVector::Ones(m), Dc = CoreVector::Ones(n);
  for (int pass=0; pass<8; ++pass) {
    if (sparse) {
      CoreVector rowmag = bs.cwiseAbs();
      for (int k=0;k<Asp.outerSize();++k)
        for (CoreSparse::InnerIterator it(Asp,k);it;++it)
          rowmag(it.row())=std::max(rowmag(it.row()),std::abs(it.value()));
      CoreVector rf = rowmag.array().max(Real(1e-12)).sqrt().inverse().max(Real(1e-3)).min(Real(1e3));
      bs.array() *= rf.array(); Dr.array() *= rf.array();
      for (int k=0;k<Asp.outerSize();++k)
        for (CoreSparse::InnerIterator it(Asp,k);it;++it) it.valueRef()*=rf(it.row());
      CoreVector colmag = cs.cwiseAbs();
      for (int k=0;k<Asp.outerSize();++k)
        for (CoreSparse::InnerIterator it(Asp,k);it;++it)
          colmag(it.col())=std::max(colmag(it.col()),std::abs(it.value()));
      CoreVector cf = colmag.array().max(Real(1e-12)).sqrt().inverse().max(Real(1e-3)).min(Real(1e3));
      cs.array() *= cf.array(); Dc.array() *= cf.array();
      for (int k=0;k<Asp.outerSize();++k)
        for (CoreSparse::InnerIterator it(Asp,k);it;++it) it.valueRef()*=cf(it.col());
      continue;
    }
    for (int i=0; i<m; ++i) {
      Real mag = std::max(As.row(i).cwiseAbs().maxCoeff(), std::abs(bs(i)));
      Real f = 1/std::sqrt(std::max(mag, Real(1e-12)));
      f = std::max(Real(1e-3), std::min(Real(1e3), f));
      As.row(i) *= f; bs(i) *= f; Dr(i) *= f;
    }
    int off=0;
    for (const auto& block : blocks) {
      int d=BlockDim(block);
      if (block.type == ConeType::Semidefinite) {
        CoreVector magnitude = cs.segment(off,d).cwiseAbs();
        if (m > 0) for (int j=0;j<d;++j)
          magnitude(j) = std::max(magnitude(j), As.col(off+j).cwiseAbs().maxCoeff());
        CoreVector vertex = magnitude.head(block.size);
        int k=block.size;
        for (int col=1;col<block.size;++col) for (int row=0;row<col;++row) {
          vertex(row)=std::max(vertex(row),magnitude(k));
          vertex(col)=std::max(vertex(col),magnitude(k)); ++k;
        }
        CoreVector factors(block.size);
        for (int j=0;j<block.size;++j) factors(j)=std::max(Real(1e-3),std::min(Real(1e3),std::pow(std::max(vertex(j),Real(1e-12)),-Real(0.25))));
        CoreVector scale(d); scale.head(block.size)=factors.array().square().matrix();
        k=block.size;
        for (int col=1;col<block.size;++col) for (int row=0;row<col;++row) scale(k++)=factors(row)*factors(col);
        As.middleCols(off,d) *= scale.asDiagonal(); cs.segment(off,d).array() *= scale.array(); Dc.segment(off,d).array() *= scale.array();
      } else {
        for (int j=0; j<d;) {
          int width = block.type == ConeType::Nonnegative ? 1 : d;
          Real mag = cs.segment(off+j,width).cwiseAbs().maxCoeff();
          if (m > 0) mag = std::max(mag, As.middleCols(off+j,width).cwiseAbs().maxCoeff());
          Real f = 1/std::sqrt(std::max(mag, Real(1e-12)));
          f = std::max(Real(1e-3), std::min(Real(1e3), f));
          As.middleCols(off+j,width) *= f; cs.segment(off+j,width) *= f; Dc.segment(off+j,width) *= f;
          j += width;
        }
      }
      off += d;
    }
  }
  const CoreVector e = CoreIdentity(blocks);
  const bool orthant = std::all_of(blocks.begin(), blocks.end(),
      [](const ConeBlock& block) { return block.type == ConeType::Nonnegative; });
  out.x = orthant ? Real(5.0) * e : e; out.s = out.x; out.y = CoreVector::Zero(m);
  out.status = Status::MaxIterations;
  for (int iter = 0; iter <= options.max_iterations; ++iter) {
    out.iterations = iter;
    if (!RepairRoundoffPSD(blocks, &out.x) || !RepairRoundoffPSD(blocks, &out.s)) {
      out.status=Status::NumericalFailure; break;
    }
    CoreVector rp = bs - (sparse ? CoreVector(Asp*out.x) : CoreVector(As*out.x));
    CoreVector rd = cs - (sparse ? CoreVector(Asp.transpose()*out.y) : CoreVector(As.transpose()*out.y)) - out.s;
    const CoreVector original_x = Dc.cwiseProduct(out.x);
    const CoreVector original_y = Dr.cwiseProduct(out.y);
    const CoreVector original_s = out.s.cwiseQuotient(Dc);
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
      CoreMatrix W, Wi;
      CoreVector w, wi, lambda;
      if (sparse) {
        // Orthant specialization of the same Jordan quadratic scaling:
        // W=P(sqrt(w))=diag(sqrt(x/s)), lambda=W*s.
        w=(out.x.array()/out.s.array()).sqrt(); wi=w.cwiseInverse();
        lambda=w.cwiseProduct(out.s);
      } else {
        NTScaling(blocks, out.x, out.s, &W, &Wi);
        lambda = W * out.s;
      }
      CoreVector l2 = CoreProduct(blocks, lambda, lambda);
      Eigen::FullPivLU<CoreMatrix> lambda_lu;
      if (!sparse) {
        lambda_lu.compute(L(blocks, lambda));
        if (!lambda_lu.isInvertible()) { out.status = Status::NumericalFailure; break; }
      }
      // Both paths solve A W u=rp, A^T dy+W^-1 v=rd, u+v=q.
      CoreMatrix K;
      std::unique_ptr<EquilibratedLU> dense_lu;
      CoreSparse normal;
      Eigen::SimplicialLDLT<CoreSparse> sparse_ldlt;
      Eigen::SparseLU<CoreSparse> sparse_lu;
      bool use_ldlt=false;
      if (sparse && m > 0) {
        CoreSparse scaled=Asp;
        for (int k=0;k<scaled.outerSize();++k)
          for (CoreSparse::InnerIterator it(scaled,k);it;++it)
            it.valueRef()*=w(it.col())*w(it.col());
        normal=scaled*Asp.transpose(); normal.makeCompressed();
        sparse_ldlt.compute(normal); use_ldlt=sparse_ldlt.info()==Eigen::Success;
        if (!use_ldlt) {
          // Regularization handles dependent equality rows. The actual Newton
          // residual is still checked below against the unregularized system.
          for (int i=0;i<m;++i) normal.coeffRef(i,i) += Real(1e-10);
          normal.makeCompressed(); sparse_lu.compute(normal);
          if (sparse_lu.info()!=Eigen::Success) { out.status=Status::NumericalFailure; break; }
        }
      } else if (!sparse) {
        K = CoreMatrix::Zero(m+2*n,m+2*n);
        K.block(0,0,m,n)=As*W; K.block(m,n,n,m)=As.transpose(); K.block(m,n+m,n,n)=Wi;
        K.block(m+n,0,n,n).setIdentity(); K.block(m+n,n+m,n,n).setIdentity();
        dense_lu=std::make_unique<EquilibratedLU>(K);
        if (!dense_lu->invertible()) { out.status=Status::NumericalFailure; break; }
      }
      auto solve_direction = [&](const CoreVector& rc, CoreVector* u,
                                 CoreVector* dy, CoreVector* v) {
        CoreVector q = sparse ? rc.cwiseQuotient(lambda) : lambda_lu.solve(rc).eval();
        if (sparse) {
          CoreVector nrhs = rp - Asp*w.cwiseProduct(q) + Asp*(w.array().square().matrix().cwiseProduct(rd));
          if (m==0) dy->resize(0);
          else if (use_ldlt) *dy=sparse_ldlt.solve(nrhs);
          else *dy=sparse_lu.solve(nrhs);
          *v=w.cwiseProduct(rd-Asp.transpose()*(*dy)); *u=q-*v;
          const Real direction_error = (Asp*w.cwiseProduct(*u)-rp).norm();
          if (std::getenv("JORDAN_TRACE")) std::cerr << "sparse direction residual=" << direction_error/(1+rp.norm()) << "\n";
          return u->allFinite() && dy->allFinite() && v->allFinite() &&
              direction_error <= Real(1e-5)*(1+rp.norm());
        }
        CoreVector rhs(m+2*n); rhs << rp,rd,q;
        CoreVector direction=dense_lu->solve(rhs);
        if (!direction.allFinite() || (K*direction-rhs).norm()>Real(1e-7)*(1+rhs.norm())) return false;
        *u=direction.head(n); *dy=direction.segment(n,m); *v=direction.tail(n); return true;
      };
      CoreVector au, ay, av;
      if (!solve_direction(-l2,&au,&ay,&av)) { out.status=Status::NumericalFailure; break; }
      CoreVector dx_aff = sparse ? w.cwiseProduct(au) : CoreVector(W*au);
      CoreVector ds_aff = sparse ? wi.cwiseProduct(av) : CoreVector(Wi*av);
      Real ap_aff = Step(blocks, out.x, dx_aff), ad_aff = Step(blocks, out.s, ds_aff);
      Real mu = out.x.dot(out.s) / e.squaredNorm();
      Real mu_aff = (out.x + ap_aff*dx_aff).dot(out.s + ad_aff*ds_aff) / e.squaredNorm();
      Real sigma = std::pow(std::max(Real(0.0), std::min(Real(1.0), mu_aff/mu)), 3);
      if (!orthant) sigma = std::max(Real(0.1), sigma);
      CoreVector correction = CoreProduct(blocks, au, av);
      CoreVector u, dy, v;
      if (!solve_direction(sigma*mu*e-l2-correction,&u,&dy,&v)) {
        out.status=Status::NumericalFailure; break;
      }
      CoreVector dx = sparse ? w.cwiseProduct(u) : CoreVector(W*u);
      CoreVector ds = sparse ? wi.cwiseProduct(v) : CoreVector(Wi*v);
      const Real fraction = orthant ? Real(0.995) : Real(0.9);
      Real ap = std::min(Real(1.0), fraction * Step(blocks, out.x, dx));
      Real ad = std::min(Real(1.0), fraction * Step(blocks, out.s, ds));
      if (std::getenv("JORDAN_TRACE")) std::cerr << "iter=" << iter << " rp=" << out.primal_residual << " rd=" << out.dual_residual << " gap=" << out.gap << " ap=" << ap << " ad=" << ad << " sigma=" << sigma << "\n";
      if (ap <= Real(1e-14) || ad <= Real(1e-14)) { out.status = Status::NumericalFailure; break; }
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

} } }

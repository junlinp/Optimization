#include "linear_programing.h"

#include <assert.h>

#include "iostream"
#include <cmath>

#include "Eigen/SVD"

namespace internal {
using namespace Eigen;

template <class T> void Unused(T v) { (void)(v); };

void Block(SparseMatrix<double> &A, int start_row, int start_col,
           int row_offset, int col_offset, SparseMatrix<double> &Rhs) {
  Unused(row_offset);
  Unused(col_offset);
  assert(start_row >= 0);
  assert(start_col >= 0);
  assert(start_row < A.rows());
  assert(start_col < A.cols());
  assert(row_offset >= 0);
  assert(col_offset >= 0);
  assert(row_offset < Rhs.rows());
  assert(col_offset < Rhs.cols());
  for (int k = 0; k < Rhs.outerSize(); k++) {
    for (SparseMatrix<double>::InnerIterator it(Rhs, k); it; ++it) {
      A.insert(start_row + it.row(), start_col + it.col()) = it.value();
    }
  }
}
std::pair<SparseMatrix<double>, Eigen::VectorXd> ToSelfEmbeddingProblem(
    const Eigen::VectorXd& c, const SparseMatrix<double>& A, const VectorXd& b) {
  using SM = SparseMatrix<double>;
  MatrixXd::Index m = A.rows();
  MatrixXd::Index n = A.cols();

  SM H_hat(m + n + 1, m + n + 1);
  //           |   0     A      -b |
  // H_hat =   |   -A^T  0       c |
  //           |    b^T  -c^T    0 |
  for(int k = 0; k < A.outerSize(); k++) {
    for (SM::InnerIterator it(A, k); it; ++it) {
      H_hat.insert(m + it.col(), 0 +  it.row()) =  -it.value();
      H_hat.insert(0 + it.row(), m + it.col()) = it.value();
    }
  }
  for(int i = 0; i < b.rows(); i++) {
    H_hat.insert(i, m + n) = -b(i);
    H_hat.insert(m + n, i) = b(i);
  }

  for(int i = 0; i < c.rows(); i++) {
    H_hat.insert(m + i, m + n) = c(i);
    H_hat.insert(m + n, m + i) = -c(i);
  }

  VectorXd r =
      VectorXd::Ones(m + n + 1, 1) - H_hat * VectorXd::Ones(m + n + 1, 1);
  SM H(m + n + 2, m + n + 2);
  for(int k = 0; k < H_hat.outerSize(); k++) {
    for(SM::InnerIterator it(H_hat, k); it; ++it) {
      H.insert(it.row(), it.col()) = it.value();
    }
  }

  for(int i = 0; i < r.rows(); i++) {
    H.insert(i, m + n + 1) = r(i);
    H.insert(m + n + 1, i) = -r(i);
  }
  VectorXd q = VectorXd::Zero(m + n + 2, 1);
  q(m + n + 1) = m + n + 2;
  return {H, q};
}
std::pair<Eigen::MatrixXd, Eigen::VectorXd> ToSelfEmbeddingProblem(
    const Eigen::VectorXd& c, const Eigen::MatrixXd& A, const VectorXd& b) {
  MatrixXd::Index m = A.rows();
  MatrixXd::Index n = A.cols();

  MatrixXd H_hat(m + n + 1, m + n + 1);
  H_hat.setZero();
  //           |   0     A      -b |
  // H_hat =   |   -A^T  0       c |
  //           |    b^T  -c^T    0 |
  H_hat.block(0, m, m, n) = A;
  H_hat.block(0, m + n, m, 1) = -b;

  H_hat.block(m, 0, n, m) = -A.transpose();
  H_hat.block(m, m + n, n, 1) = c;

  H_hat.block(m + n, 0, 1, m) = b.transpose();
  H_hat.block(m + n, m, 1, n) = -c.transpose();

  VectorXd r =
      VectorXd::Ones(m + n + 1, 1) - H_hat * VectorXd::Ones(m + n + 1, 1);
  MatrixXd H(m + n + 2, m + n + 2);
  H.setZero();

  H.block(0, 0, m + n + 1, m + n + 1) = H_hat;
  H.block(0, m + n + 1, m + n + 1, 1) = r;
  H.block(m + n + 1, 0, 1, m + n + 1) = -r.transpose();

  VectorXd q = VectorXd::Zero(m + n + 2, 1);
  q(m + n + 1) = m + n + 2;
  return {H, q};
}
    SparseMatrix<double> SwapPermutation(size_t i, size_t j, size_t n) {
        SparseMatrix<double> res(n, n);
        for(int k = 0; k < n; k++) {
          if (k != i && k != j) {
            res.insert(k, k) = 1;
          }
        }
        res.insert(i, j) = 1;
        res.insert(j, i) = 1;
        return res;
    }
    /**
     @breif Compute a colmun permutation for a  m x n matrix A
     such that the first m colum is full rank.
     */
    MatrixXd Permutation(const MatrixXd& A) {
        MatrixXd::Index m = A.rows(), n = A.cols();
        assert(m <= n);
        assert(m > 0);
        assert(n > 0);
        std::vector<VectorXd> base_vector;
        std::vector<size_t> base_index;
        base_vector.reserve(m);
        base_index.reserve(m);
        base_vector.push_back(A.col(0).normalized());
        base_index.push_back(0);
        for (int i = 1; i < n; i++) {
            if (base_vector.size() == size_t(m)) {
                break;
            }
            VectorXd current = A.col(i);
            for(const auto& base : base_vector) {
                    current = current - current.dot(base) * base;
            }
            if (current.norm() > n * 1e-6) {
                base_vector.push_back(current.normalized());
                base_index.push_back(i);
            }
        }
        
        if (base_index.size() == size_t(m)) {
           MatrixXd res = MatrixXd::Identity(n, n);
            for(int i = 0; i < m; i++) {
                res = res * MatrixXd(SwapPermutation(i, base_index[i], n));
            }
            return res;
        } else {
            // error
            throw std::invalid_argument("Matrix is not full rank");
        }
    }

    SparseMatrix<double> Permutation(const SparseMatrix<double>& A) {
        MatrixXd::Index m = A.rows(), n = A.cols();
        assert(m <= n);
        assert(m > 0);
        assert(n > 0);
        std::vector<SparseVector<double>> base_vector;
        std::vector<size_t> base_index;
        base_vector.reserve(m);
        base_index.reserve(m);
        SparseVector<double> init = A.col(0) / A.col(0).norm();
        base_vector.push_back(init);
        base_index.push_back(0);
        std::cout << "Permutation m = " << m << ", n = " << n << std::endl;
        for (int i = 1; i < n; i++) {
            if (base_vector.size() == size_t(m)) {
                break;
            }
            SparseVector<double> current = A.col(i);

            for(const auto& base : base_vector) {
                    current = current - current.dot(base) * base;
            }

            if (current.norm() > n * 1e-6) {
                SparseVector<double> t = current / current.norm();
                base_vector.push_back(std::move(t));
                base_index.push_back(i);
            }
        }
        
        if (base_index.size() == size_t(m)) {
           SparseMatrix<double> res(n, n); 
           for(int k = 0; k < n; k++) {
            res.insert(k, k) = 1.0;
           }
            for(int i = 0; i < m; i++) {
                res = res * SparseMatrix<double>(SwapPermutation(i, base_index[i], n));
            }
            return res;
        } else {
            // error
            throw std::invalid_argument("Matrix is not full rank");
        }
    }
};

template<class Matrix, class Vector>
void ReduceInEqual(Matrix& A, Vector& a, const Matrix& B,const Vector& b) {
  assert(A.cols() == B.cols());

  int total_row_size = A.rows() + B.rows();
  int column_size = A.cols();
  Matrix res(total_row_size, column_size);
  Vector res_b(total_row_size);

  for (int k=0; k< A.outerSize(); ++k) {
    for (typename Matrix::InnerIterator it(A, k); it; ++it) {
      it.value();
      it.row();  // row index
      it.col();  // col index (here it is equal to k)
      res.insert(it.row(), it.col()) = it.value();
    }
  }

  for (typename Vector::InnerIterator it(a); it; ++it) {
    res_b.insert(it.index()) = it.value();
  }

  for (int k=0; k< B.outerSize(); ++k) {
    for (typename Matrix::InnerIterator it(B, k); it; ++it) {
      it.value();
      it.row();  // row index
      it.col();  // col index (here it is equal to k)
      res.insert(A.rows() + it.row(), it.col()) = it.value();
    }
  }

  for (typename Vector::InnerIterator it(b); it; ++it) {
    res_b.insert(a.rows() + it.index()) = it.value();
  }

  A = res;
  a = res_b;


}
/*
int ConstructProblem(Problem& problem, Eigen::VectorXd& c, Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b) {
  using SM = Eigen::SparseMatrix<double, Eigen::RowMajor>;

  SM M(problem.row_offset.size(), problem.column_offset.size());
  std::vector<Eigen::Triplet<double>> triple;
  triple.reserve(problem.row_column_value.size());
  for(auto&& [row_name, column_name, value] : problem.row_column_value) {
    int row_index = problem.row_offset[row_name];
    int column_index = problem.column_offset[column_name];
    triple.emplace_back(row_index, column_index, value);
  }
  M.setFromTriplets(triple.begin(), triple.end());
  size_t n = M.cols();
  std::cout << "M Has : " << M.nonZeros() << " nnz" << std::endl;
  using SV = Eigen::SparseVector<double>;

  SV RHS(M.rows());
  for (auto&& [row_name, value] : problem.rhs) {
    int row_index = problem.row_offset[row_name];
    RHS.insert(row_index) = value;
  }
  std::cout << "RHS Has " << RHS.nonZeros() << " nnz" << std::endl;

  SM UPBounds(problem.up_bounds.size(), n);
  SV UPBounds_b(problem.up_bounds.size());
  int count = 0;
  for ( auto&&[column_name, value] : problem.up_bounds) {
    int column_index = problem.column_offset[column_name];
    UPBounds.insert(count, column_index) = 1.0;
    UPBounds_b.insert(count) = value;
    count++;
  }
  
  SM LOBounds(problem.lower_bounds.size(), n);
  SV LOBounds_b(problem.lower_bounds.size());
  count = 0;
  for (auto&& [column_name, value] : problem.lower_bounds) {
    int column_index = problem.column_offset[column_name];
    LOBounds.insert(count, column_index) = 1.0;
    LOBounds_b.insert(count) = value;
    count++;
  }
  c = M.row(0);
  SM E = M.middleRows(1, problem.equal_size);
  SV Eb = RHS.middleRows(1, problem.equal_size);

  SM L = M.middleRows(1 + problem.equal_size, problem.less_size);
  SV Lb = RHS.middleRows(1 + problem.equal_size, problem.less_size);

  SM G = M.middleRows(1 + problem.equal_size + problem.less_size, problem.greater_size);
  SV Gb = RHS.middleRows(1 + problem.equal_size + problem.less_size, problem.greater_size);

  SM IA = -L;
  SV Ib = -Lb;
  // IA * x >= Ib
  ReduceInEqual(IA, Ib, G, Gb);
  ReduceInEqual(IA, Ib, SM(-UPBounds), SV(-UPBounds_b));
  ReduceInEqual(IA, Ib, LOBounds, LOBounds_b);

  int variable_size = E.cols();
  int m1 = E.rows(), m2 = IA.rows();
  Eigen::VectorXd c_dot(2 * variable_size + m2);
  c_dot << c, -c, Eigen::VectorXd::Zero(m2);
  c = c_dot;
  A = Eigen::SparseMatrix<double>(m1 + m2, variable_size * 2 + m2);

  std::vector<Eigen::Triplet<double>> A_triple;

  for (int k = 0; k < E.outerSize(); k++) {
    for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(E, k); it; ++it) {
      int row = it.row();
      int col = it.col();
      double value = it.value();
      A_triple.emplace_back(row, col, value);
      A_triple.emplace_back(row, col + variable_size, -value);
    }
  }

  for (int k = 0; k < IA.outerSize(); k++) {
    for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(IA,k ); it; ++it) {
      int row = it.row();
      int col = it.col();
      double value = it.value();      

      A_triple.emplace_back(row + m1, col, value);
      A_triple.emplace_back(row + m1, col + variable_size, -value);
    }
  }

  for(int i = 0; i < m2; i++) {
    A_triple.emplace_back(m1 + i, 2 * variable_size+ i, 1.0);
  }

  A.setFromTriplets(A_triple.begin(), A_triple.end());

  b = Eigen::VectorXd(E.rows() + IA.rows());
  b << Eigen::VectorXd(Eb), Eigen::VectorXd(Ib);


  // min <c, x>
  // s.t Ex = Eb
  //     Lb <= Lx
  //      0 <= Gx <= Gb

  //  min <c, x1> - <c, x2>
  // s.t
  //   Ax1 - Ax2 + 0y = b;
  //   Gx1 - Gx2 + Iy = d
  //   y >= 0
  //   x1, x2 >= 0
  //   

  // min <c', z>
  // s.t 
  //      Mz = q
  //       z >= 0
  //            n1  n1
  //  where M = A  -A  0
  //            G  -G  I
  //        z= [x1, x2, y]
  //        c' = [c, -c, 0]
  //        q = [b, d];

  return 1;
}
*/

/**
 @breif Compute LO Problem with DualLogarithm method.
 we will convert the Standard-Form Linear Programing probem to self-embeding problem.
 the Standard-Form Problem (P):
 min   <c, x>
 s.t    Ax=b
        x> 0
 
 we will permutation the matrix A thus A can be deconpose as A = [Ai, Aj] which Ai is a m x m matrix
 with full rank. then Ax = b can be wrote as Ai * xi + Aj * xj = b.
 so xi = inv(Ai) * (b - Aj * xj) > 0
 <c, x> = <cj, xj> + <ci, xi> = <ci, inv(Ai) * (b - Aj * xj)> + <cj, xj> = <ci, inv(Ai) * b> - <ci, inv(Ai) * Aj * xj> + <cj, xj>
 = <cj - Aj^T * inv(Ai)^T * ci, xj> +<ci, inv(Ai) * b>
 
 min <cj - Aj^T * inv(Ai)^T * ci, xj>
 s.t inv(Ai) * (b - Aj * xj) > 0
       xj > 0
 
 
 */

template<class T>
T linsolve(const Eigen::MatrixXd& A, const T& b) {
        return T{A.fullPivHouseholderQr().solve(b)};
}
template< class RHS>
RHS linsolve(const Eigen::SparseMatrix<double>& A, const RHS& b) {
  Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
  solver.compute(A);
  return solver.solve(b);
}

Eigen::VectorXd FeasibleDualLogarithmSolver(const Eigen::VectorXd& c, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initial_x, const Eigen::VectorXd& initial_y, const Eigen::VectorXd& initial_s) {
  internal::Unused(initial_y);
  internal::Unused(c);
  internal::Unused(b);
    // H is the null space of A
    size_t m = A.rows(), n = A.cols();
    Eigen::MatrixXd V = A.bdcSvd(Eigen::ComputeFullV).matrixV();
    
    Eigen::MatrixXd H = V.block(0, m, n, n - m).transpose();
    Eigen::VectorXd e(n);
    e.setOnes();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    
    double theta = 1.0 / (3 * std::sqrt(n));
    double mu = 1.0;
    double epsilon = 1e-10;
    Eigen::VectorXd s = initial_s;
    size_t epoch  = 0;
    while (n * mu >= epsilon) {
        Eigen::MatrixXd S = s.asDiagonal();
        Eigen::MatrixXd t = linsolve<Eigen::MatrixXd>(H * S * S * H.transpose(), H * S);
        Eigen::VectorXd delta_s = S * (I - S * H.transpose() * t) * (e - S*initial_x / mu);
        mu = (1 - theta) * mu;
        s = s + delta_s;
        epoch ++;
    }
    std::printf("Feasible Dual Logarithm Method %zu epochs\n", epoch);
    return s.cwiseInverse() * mu;
}

Eigen::MatrixXd ProjectOperator(const Eigen::MatrixXd& A) {
    size_t n = A.cols();

    Eigen::MatrixXd I(n, n);
    I.setIdentity();

    return I - A.transpose() * (A * A.transpose()).fullPivLu().solve(A);
}

Eigen::VectorXd FeasibleDualLogarithmWithAdaptiveUpdateSolver(const Eigen::VectorXd& c, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initial_x, const Eigen::VectorXd& initial_y, const Eigen::VectorXd& initial_s) {
  internal::Unused(initial_y);
  internal::Unused(b);
  internal::Unused(c);
    // H is the null space of A
    size_t m = A.rows(), n = A.cols();
    Eigen::MatrixXd V = A.bdcSvd(Eigen::ComputeFullV).matrixV();
    Eigen::MatrixXd H = V.block(0, m, n, n - m).transpose();
    Eigen::VectorXd e(n);
    e.setOnes();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    
    double theta = 1.0 / (3 * std::sqrt(n));
    internal::Unused(theta);
    double mu = 1.0;
    double epsilon = 1e-10;
    double tau = 1.0 / std::sqrt(2);
    Eigen::VectorXd s = initial_s;
    size_t epoch  = 0;
    while (n * mu >= epsilon) {
        Eigen::MatrixXd S = s.asDiagonal();
        //Eigen::MatrixXd t = linsolve<Eigen::MatrixXd>(H * S * S * H.transpose(), H * S);
        //Eigen::MatrixXd Project = (I - S * H.transpose() * t);
        Eigen::MatrixXd Project = ProjectOperator(H*S);
        //Eigen::VectorXd delta_s = S *  * (e - S*initial_x / mu);
        Eigen::VectorXd dc = Project * e;
        Eigen::VectorXd da = -Project * S * initial_x;
        Eigen::VectorXd delta_c_s = S * dc;
        Eigen::VectorXd delta_a_s = S * da;
        s = s + delta_c_s + delta_a_s / mu;
        std::cout << "S : " << s << std::endl;
        double dc_dot_da = da.dot(dc);

        double p = dc_dot_da + sqrt(dc_dot_da * dc_dot_da - da.squaredNorm() * (dc.squaredNorm() - tau * tau));
        double temp_mu = (dc.squaredNorm() - tau * tau) / p;

        std::printf("mu = %f, new_mu = %f\n", mu, temp_mu);
        //mu = (1 - theta) * mu;
        mu = temp_mu;
        epoch ++;
    }
    std::printf("Feasible Dual Logarithm Adaptive Method %zu epochs\n", epoch);
    return s.cwiseInverse() * mu;
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> KKTSolver(const Eigen::MatrixXd& A, const Eigen::VectorXd& center) {
  // solve   Ax = 0
  //         A^T*y + s = 0
  //           x + s = center

  Eigen::VectorXd y = linsolve<Eigen::VectorXd>(A * A.transpose(), -A * center);
  Eigen::VectorXd s = - A.transpose() * y;
  Eigen::VectorXd x = center + A.transpose() * y;
  return {x, y, s};
}

Eigen::VectorXd FeasiblePrimDualLogarithmSolver(
    const Eigen::VectorXd &c, const Eigen::MatrixXd &A,
    const Eigen::VectorXd &b, const Eigen::VectorXd &initial_x,
    const Eigen::VectorXd &initial_y, const Eigen::VectorXd &initial_s) {
  internal::Unused(initial_y);
  internal::Unused(c);

  size_t m = A.rows(), n = A.cols();

  //Eigen::MatrixXd V = A.bdcSvd(Eigen::ComputeFullV).matrixV();
  //Eigen::MatrixXd H = V.block(0, m, n, n - m).transpose();

  double theta = 1.0 / std::sqrt(2 * n);
  double epsilon = 1e-10;
  double mu = 1.0;
  Eigen::VectorXd x = initial_x, y = initial_y, s = initial_s;
  size_t epoch = 0;
  while (mu * n >= epsilon ){
    Eigen::VectorXd inv_s = s.cwiseInverse();
    Eigen::VectorXd delta_y = linsolve<Eigen::VectorXd>(A * x.asDiagonal() * inv_s.asDiagonal() * A.transpose(), b - mu * A * inv_s);
    Eigen::VectorXd delta_s = -A.transpose() * delta_y;
    Eigen::VectorXd delta_x = mu * inv_s - x - x.asDiagonal() * (inv_s.asDiagonal() * delta_s);
    //Eigen::VectorXd u = x.cwiseProduct(s).cwiseSqrt() / mu;
    //Eigen::VectorXd d = (x.cwiseProduct(s.cwiseInverse())).cwiseSqrt();
    //Eigen::MatrixXd PAD = ProjectOperator(A * d.asDiagonal());
    //Eigen::MatrixXd PHDinv = ProjectOperator(H * d.cwiseInverse().asDiagonal());
    //Eigen::VectorXd dx = PAD * (u.cwiseInverse() - u);
    //Eigen::VectorXd ds = PHDinv * (u.cwiseInverse() - u);

    //auto [delta_x, delta_y, delta_s] = KKTS
    //x += sqrt(mu) * d.asDiagonal() * dx;
    y += delta_y;
    //s += sqrt(mu) * d.cwiseInverse().asDiagonal() * ds;
    x += delta_x;
    s += delta_s;
    mu = (1 - theta ) * mu;
    epoch++;
  }
  std::printf("Feasible Prim Dual Logarithm Method %zu epochs\n", epoch);
  return x;
}

Eigen::VectorXd FeasiblePrimDualLogarithmSolver(const Eigen::VectorXd& c, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initial_x, const Eigen::VectorXd& initial_y, const Eigen::VectorXd& initial_s) {
  internal::Unused(initial_y);
  internal::Unused(c);

  size_t m = A.rows(), n = A.cols();
  double theta = 1.0 / std::sqrt(2 * n);
  double epsilon = 1e-10;
  double mu = 1.0;
  Eigen::VectorXd x = initial_x, y = initial_y, s = initial_s;
  size_t epoch = 0;
  while (mu * n >= epsilon ){
    Eigen::VectorXd inv_s = s.cwiseInverse();
    Eigen::VectorXd delta_y = linsolve<Eigen::VectorXd>(Eigen::SparseMatrix<double>(A * x.asDiagonal() * inv_s.asDiagonal() * A.transpose()), b - mu * A * inv_s);
    Eigen::VectorXd delta_s = -A.transpose() * delta_y;
    Eigen::VectorXd delta_x = mu * inv_s - x - x.asDiagonal() * (inv_s.asDiagonal() * delta_s);
    y += delta_y;
    x += delta_x;
    s += delta_s;
    mu = (1 - theta ) * mu;
    epoch++;

    if (epoch % 5 == 0) {
      std::printf("mu * n = %.13f\n", mu * n);
    }
  }
  std::printf("Feasible Prim Dual Logarithm Method %zu epochs\n", epoch);
  return x;
}

Eigen::VectorXd FeasiblePrimDualLogarithmPredictorCorrectorSolver(const Eigen::VectorXd& c, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initial_x, const Eigen::VectorXd& initial_y, const Eigen::VectorXd& initial_s) {
  internal::Unused(c);

  size_t m = A.rows(), n = A.cols();
  internal::Unused(m);

  double theta = 1.0 / std::sqrt(2 * n);
  double epsilon = 1e-10;
  double mu = 1.0;
  Eigen::VectorXd x = initial_x, y = initial_y, s = initial_s;
  size_t epoch = 0;
  while (mu * n >= epsilon ){
    Eigen::VectorXd inv_s = s.cwiseInverse();
    Eigen::VectorXd delta_y = linsolve<Eigen::VectorXd>(A * x.asDiagonal() * inv_s.asDiagonal() * A.transpose(), b - mu * A * inv_s);
    Eigen::VectorXd delta_s = -A.transpose() * delta_y;
    Eigen::VectorXd delta_x = mu * inv_s - x - x.asDiagonal() * (inv_s.asDiagonal() * delta_s);

    x += delta_x;
    y += delta_y;
    s += delta_s;

    inv_s = s.cwiseInverse();


    // Compute the Correcter Direction
    //
    // compute the theta for correcter direction which is feasible

    Eigen::VectorXd affine_y = linsolve<Eigen::VectorXd>(x.asDiagonal() * A.transpose(), x.asDiagonal() * s);
    Eigen::VectorXd affine_s = -A.transpose() * affine_y;
    Eigen::VectorXd affine_x = x.asDiagonal() * (affine_s - s);

    //double norm_xs=  (affine_x.asDiagonal() * affine_s).norm();
    //theta = 2.0 / (1 + sqrt(1 + 13 / n / mu * norm_xs ));

    // how to update y
    x += theta * affine_x;
    y += theta * affine_y;
    s += theta * affine_s;
    mu = (1 - theta ) * mu;
    epoch++;
  }
  std::printf("Feasible Prim Dual Logarithm Method %zu epochs\n", epoch);
  return x;
}

template<class Matrix, class Functor>
void FeasibleSolver(const Eigen::VectorXd& c, const Matrix& A, const Eigen::VectorXd& b, Eigen::VectorXd& x, Functor functor) {
    size_t m = A.rows(), n = A.cols();

    Eigen::MatrixXd permutation_matrix = internal::Permutation(A);

    Eigen::MatrixXd A_ = (A * permutation_matrix).eval();
    Eigen::MatrixXd Ai = A_.block(0, 0, m, m);
    Eigen::MatrixXd Aj = A_.block(0, m, m, n - m);
    Eigen::VectorXd ci = c.block(0,0, m, 1);
    Eigen::VectorXd cj = c.block(m, 0, n - m, 1);

    Eigen::VectorXd self_embedding_c = cj - Aj.transpose() * linsolve(Ai.transpose(), ci);
    Eigen::MatrixXd self_embeding_A = - linsolve(Ai, Aj);
    Eigen::VectorXd self_embedding_b = - linsolve(Ai, b);
    
    auto [H, q] = internal::ToSelfEmbeddingProblem(self_embedding_c, self_embeding_A, self_embedding_b);
    
    // min <q, ksi>
    // s.t H*z + q = s
    //       z >= 0, s >= 0
    //     z = mu*e and s = mu*e is the initial feasible solution
    //
    size_t feasible_n = H.rows();
    Eigen::MatrixXd A_feasible(feasible_n, feasible_n * 2);
    A_feasible.block(0, 0, feasible_n, feasible_n) = H;
    A_feasible.block(0, feasible_n, feasible_n, feasible_n) = -Eigen::MatrixXd::Identity(feasible_n, feasible_n);
    
    Eigen::VectorXd c_feasible(feasible_n * 2);
    c_feasible << q, Eigen::VectorXd::Zero(feasible_n, 1);
    
    Eigen::VectorXd b_feasible = -q;
    
    Eigen::VectorXd x_feasible = Eigen::VectorXd::Ones(feasible_n * 2, 1);
    Eigen::VectorXd s_feasible = Eigen::VectorXd::Ones(feasible_n * 2, 1);
    Eigen::VectorXd y_feasible = Eigen::VectorXd::Ones(feasible_n, 1);
    
    Eigen::VectorXd solution = functor(c_feasible, A_feasible, b_feasible, x_feasible, y_feasible, s_feasible);
    Eigen::VectorXd ksi = solution.block(0, 0, feasible_n, 1);
    
    Eigen::VectorXd y = ksi.block(0, 0, m, 1);
    Eigen::VectorXd xj = ksi.block(m, 0, n - m, 1);
    double k = ksi(n);
    
    // TODO(junlinp): check whether is solvable.
    // xj = n - m
    // Aj = m
    
    xj = xj / k;
    Eigen::VectorXd xi = linsolve<Eigen::VectorXd>(Ai, b - Aj * xj);
    x << xi, xj;
    x = permutation_matrix * x;
}

template <class Functor>
void FeasibleSolver(const Eigen::VectorXd& c,
                    const Eigen::SparseMatrix<double>& A,
                    const Eigen::VectorXd& b, Eigen::VectorXd& x,
                    Functor functor) {
  size_t m = A.rows(), n = A.cols();
  std::cout << "Compute permutation" << std::endl;
  Eigen::SparseMatrix<double> permutation_matrix = internal::Permutation(A);
  std::cout << "Compute permutation finish" << std::endl;
  Eigen::SparseMatrix<double> A_ = (A * permutation_matrix).eval();
  Eigen::SparseMatrix<double> Ai = A_.leftCols(m);
  Eigen::SparseMatrix<double> Aj = A_.rightCols(n - m);
  Eigen::VectorXd ci = c.block(0, 0, m, 1);
  Eigen::VectorXd cj = c.block(m, 0, n - m, 1);

  Eigen::VectorXd self_embedding_c =
      cj - Aj.transpose() * linsolve(Eigen::SparseMatrix<double>(Ai.transpose()), ci);
  Eigen::SparseMatrix<double> self_embeding_A = -linsolve(Ai, Aj);
  Eigen::VectorXd self_embedding_b = -linsolve(Ai, b);

  std::cout << "Self Embedding Problem" << std::endl;
  auto [H, q] = internal::ToSelfEmbeddingProblem(
      self_embedding_c, self_embeding_A, self_embedding_b);

  // min <q, ksi>
  // s.t H*z + q = s
  //       z >= 0, s >= 0
  //     z = mu*e and s = mu*e is the initial feasible solution
  //
  size_t feasible_n = H.rows();
  Eigen::SparseMatrix<double> A_feasible(feasible_n, feasible_n * 2);

  internal::Block(A_feasible, 0, 0, feasible_n, feasible_n, H);
  for(int i = 0; i < feasible_n; i++) {
    A_feasible.insert(i, i + feasible_n) = 1.0;
  }
  //A_feasible.block(0, 0, feasible_n, feasible_n) = H;
  //A_feasible.block(0, feasible_n, feasible_n, feasible_n) =
  //    -Eigen::MatrixXd::Identity(feasible_n, feasible_n);

  Eigen::VectorXd c_feasible(feasible_n * 2);
  c_feasible << q, Eigen::VectorXd::Zero(feasible_n, 1);

  Eigen::VectorXd b_feasible = -q;

  Eigen::VectorXd x_feasible = Eigen::VectorXd::Ones(feasible_n * 2, 1);
  Eigen::VectorXd s_feasible = Eigen::VectorXd::Ones(feasible_n * 2, 1);
  Eigen::VectorXd y_feasible = Eigen::VectorXd::Ones(feasible_n, 1);
  std::cout << "Solver Implement " << std::endl;
  Eigen::VectorXd solution = functor(c_feasible, A_feasible, b_feasible,
                                     x_feasible, y_feasible, s_feasible);
  Eigen::VectorXd ksi = solution.block(0, 0, feasible_n, 1);

  Eigen::VectorXd y = ksi.block(0, 0, m, 1);
  Eigen::VectorXd xj = ksi.block(m, 0, n - m, 1);
  double k = ksi(n);

  // TODO(junlinp): check whether is solvable.
  // xj = n - m
  // Aj = m

  xj = xj / k;
  Eigen::VectorXd xi = linsolve<Eigen::VectorXd>(Ai, b - Aj * xj);
  x << xi, xj;
  x = permutation_matrix * x;
}

void DualLogarithmSolver(const Eigen::VectorXd& c, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::VectorXd& x) {
    FeasibleSolver(c, A, b, x, FeasibleDualLogarithmSolver);
    //FeasibleSolver(c, A, b, x, FeasibleDualLogarithmWithAdaptiveUpdateSolver);
}


void PrimDualLogarithmSolver(const Eigen::VectorXd& c, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::VectorXd& x) {
  //FeasibleSolver(c, A, b, x, FeasiblePrimDualLogarithmSolver);

  //FeasibleSolver(c, A, b, x, FeasiblePrimDualLogarithmPredictorCorrectorSolver);
}

void SparsePrimDualLogarithmSolver(const Eigen::VectorXd& c, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, Eigen::VectorXd& x) {

  auto functor =  [](const Eigen::VectorXd& c, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initial_x, const Eigen::VectorXd& initial_y, const Eigen::VectorXd& initial_s) ->Eigen::MatrixXd{
  return FeasiblePrimDualLogarithmSolver(c, A, b,initial_x, initial_y, initial_s);
  };
  FeasibleSolver(c, A, b, x,functor);
}


void LPSolver(const Eigen::VectorXd& c, const Eigen::MatrixXd& A,
              const Eigen::VectorXd& b, Eigen::VectorXd& x) {
  size_t n = c.rows();
  size_t m = A.rows();

  x = Eigen::VectorXd::Ones(n);
  Eigen::VectorXd z = Eigen::VectorXd::Ones(n);
  Eigen::VectorXd y = Eigen::VectorXd::Zero(m);
  double eps_feas = 1e-5;
  double eps = 1e-8;
  size_t max_iterator = 1024;
  double mu = 10.0;
  double sigma = 0.9;
  Eigen::MatrixXd zero_m_m(m, m), zero_m_n(m, n), zero_n_n(n, n);
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < m; j++) {
      zero_m_m(i, j) = 0.0;
    }
    for (size_t k = 0; k < n; k++) {
      zero_m_n(i, k) = 0.0;
    }
  }

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      zero_n_n(i, j) = 0.0;
    }
  }

  Eigen::VectorXd e = Eigen::VectorXd::Ones(n);
  std::cout << "Iterator\t\tPrimal\t\tDual" << std::endl;
  for (size_t iter = 0; iter < max_iterator; iter++) {
    Eigen::MatrixXd Z = z.asDiagonal();
    Eigen::MatrixXd X = x.asDiagonal();
    Eigen::MatrixXd H(2 * n + m, 2 * n + m);
    H << A, zero_m_m, zero_m_n, zero_n_n, A.transpose(),
        Eigen::MatrixXd::Identity(n, n), Z, zero_m_n.transpose(), X;
    // std::cout << "H : " << H << std::endl;
    Eigen::VectorXd B(2 * n + m);
    double t = (1 - sigma / std::sqrt(n)) * x.dot(z) / n;
    B << b - A * x, c - z - A.transpose() * y, t * e - X * Z * e;

    Eigen::VectorXd delta = H.fullPivLu().solve(B);
    // std::cout << "delta : " << delta << std::endl;
    Eigen::VectorXd delta_x = delta.block(0, 0, n, 1);
    Eigen::VectorXd delta_y = delta.block(n, 0, m, 1);
    Eigen::VectorXd delta_z = delta.block(n + m, 0, n, 1);

    double s_max = 1.0;
    for (size_t i = 0; i < n; i++) {
      if (delta_x(i) < 0) {
        s_max = std::min(s_max, x(i) / -delta_x(i));
      }

      if (delta_z(i) < 0) {
        s_max = std::min(s_max, z(i) / -delta_z(i));
      }
    }
    // std::cout << "s_max : " << s_max << std::endl;
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd;
    // svd.compute(X.inverse() * x.asDiagonal());
    // std::cout << "Largest EigenValue : " << svd.singularValues().maxCoeff()
    // << std::endl; Eigen::JacobiSVD<Eigen::MatrixXd> svd2;
    // svd2.compute(Z.inverse() * z.asDiagonal());
    // s_max = 1.0 / std::max(svd.singularValues().maxCoeff(),
    // svd2.singularValues().maxCoeff());
    double s = 0.95 * s_max;

    // std::cout << "Step : " << s << std::endl;
    x = x + s * delta_x;
    y = y + s * delta_y;
    z = z + s * delta_z;
    // std::cout << "Function : " << c.dot(x) << std::endl;
    mu *= 0.01;
    std::cout << iter << "\t\t" << (A * x - b).norm() << "\t\t"
              << (c - z - A.transpose() * y).norm() << std::endl;
    if ((A * x - b).norm() <= eps_feas &&
        (c - z - A.transpose() * y).norm() <= eps_feas && (x.dot(z) < eps)) {
      std::cout << "Minimum Found" << std::endl;
      break;
    }
  }
}

Eigen::VectorXd vec(const Eigen::MatrixXd& A) {
  int n = A.cols();
  int m = A.rows();
  Eigen::VectorXd res(m * n);
  for (int i = 0; i < n; i++) {
    res.block(m * i, 0, m, 1) = A.col(i);
  }
  return res;
}

Eigen::VectorXd svec(const Eigen::MatrixXd& A) {
  assert(A.rows() == A.cols());
  const int n = A.rows();
  const int vector_dimension = (n * (n + 1)) >> 1;
  Eigen::VectorXd res(vector_dimension);

  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      int vector_index = ((2 * n + 1 - i) * i) + j;
      res(vector_index) = A(i, j);
    }
  }
  return res;
}
/**
 *  tr(A * X) can be represent as the inter product of L(A) and svec(X)
 *
 */
Eigen::VectorXd L(const Eigen::MatrixXd& A) {
  assert(A.rows() == A.cols());
  const int n = A.rows();
  const int vector_dimension = (n * (n + 1)) >> 1;
  Eigen::VectorXd res(vector_dimension);

  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      int vector_index = ((2 * n + 1 - i) * i) + j;
      res(vector_index) = A(i, j) + A(j, i);
    }
  }
  return res;
}

Eigen::MatrixXd KroneckerProduct(const Eigen::MatrixXd& A,
                                 const Eigen::MatrixXd& B) {
  Eigen::MatrixXd res(A.rows() * B.rows(), A.cols() * B.cols());

  for (int i = 0; i < A.rows(); i++) {
    for (int j = 0; j < A.cols(); j++) {
      res.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
    }
  }
  return res;
}
Eigen::MatrixXd KroneckerSum(const Eigen::MatrixXd& A,
                             const Eigen::MatrixXd& B) {
  return KroneckerProduct(A, B) + KroneckerProduct(B, A);
}

Eigen::MatrixXd mat(const Eigen::VectorXd& vec) {
  size_t n = static_cast<size_t>(sqrtf(vec.rows()) + 0.5);
  Eigen::MatrixXd res(n, n);

  for (size_t i = 0; i < n; i++) {
    res.col(i) = vec.block(n * i, 0, n, 1);
  }
  return res;
}
double ComputeMinimumEigenValue(const Eigen::MatrixXd& Matrix) {
  Eigen::EigenSolver<Eigen::MatrixXd> eigen(Matrix);
  // std::cout << "singular : " << singular << std::endl;
  auto eigen_value = eigen.eigenvalues();
  return eigen_value(eigen_value.rows() - 1).real();
}

double ComputeLargestSingularValue(const Eigen::MatrixXd& Matrix) {
  Eigen::BDCSVD<Eigen::MatrixXd> svd;
  svd.compute(Matrix);
  auto singular = svd.singularValues();
  return singular(0);
}
void SymmetricSolver(const Eigen::MatrixXd C,
                     const std::vector<Eigen::MatrixXd>& A,
                     const Eigen::VectorXd& b, Eigen::MatrixXd& x) {
  // assume C is a square matrix
  assert(C.rows() == C.cols());
  const int n = C.rows();
  const int m = A.size();
  Eigen::MatrixXd A_(m, n * n);
  for (int i = 0; i < m; i++) {
    A_.row(i) = vec(A[i]);
  }
  double mu = 0.1;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  x = I;
  Eigen::MatrixXd z = I;
  Eigen::VectorXd y = Eigen::VectorXd::Zero(m);
  Eigen::MatrixXd zero_m_m(m, m), zero_m_n(m, n * n), zero_n_n(n * n, n * n);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      zero_m_m(i, j) = 0.0;
    }
    for (int k = 0; k < n * n; k++) {
      zero_m_n(i, k) = 0.0;
    }
  }

  for (int i = 0; i < n * n; i++) {
    for (int j = 0; j < n * n; j++) {
      zero_n_n(i, j) = 0.0;
    }
  }
  const size_t max_iterator = 16;
  for (size_t iterator = 0; iterator < max_iterator; iterator++) {
    Eigen::MatrixXd X = x;
    Eigen::MatrixXd Z = z;
    x = X;
    Eigen::MatrixXd Jacobian(m + 2 * n * n, m + 2 * n * n);
    Eigen::VectorXd g(m + 2 * n * n);
    Jacobian << A_, zero_m_m, zero_m_n, zero_n_n, A_.transpose(),
        Eigen::MatrixXd::Identity(n * n, n * n),
        KroneckerProduct(Z, I) + KroneckerProduct(I, Z), zero_m_n.transpose(),
        KroneckerProduct(X, I) + KroneckerProduct(I, X);

    g << b - A_ * vec(X), vec(C) - vec(Z) - A_.transpose() * y,
        vec(2 * mu * I - X * Z - Z * X);

    Eigen::VectorXd delta = Jacobian.fullPivLu().solve(g);
    Eigen::VectorXd delta_x = delta.block(0, 0, n * n, 1);
    Eigen::VectorXd delta_y = delta.block(n * n, 0, m, 1);
    Eigen::VectorXd delta_z = delta.block(n * n + m, 0, n * n, 1);

    double s_a = 1.0;
    while (ComputeMinimumEigenValue(x + s_a * mat(delta_x)) < 1e-6) {
      s_a *= 0.9;
    }
    double s_b = 1.0;
    while (ComputeMinimumEigenValue(z + s_b * mat(delta_z)) < 1e-6) {
      s_b *= 0.9;
    }
    std::cout << "Largest X : " << s_a << std::endl;
    std::cout << "Largest Z : " << s_b << std::endl;
    // std::cout << "x + s * mat(delta_x) : " << ComputeMinimumSingularValue(x +
    // s * mat(delta_x)) << std::endl;

    std::cout << s_a << std::endl;
    std::cout << s_b << std::endl;
    x = x + s_a * mat(delta_x);
    y = y + s_b * delta_y;
    z = z + s_b * mat(delta_z);
    Eigen::BDCSVD<Eigen::MatrixXd> svd;
    svd.compute(x);
    std::cout << "singular : " << svd.singularValues() << std::endl;
    std::cout << "X : " << x << std::endl;
    std::cout << iterator << "\t\t" << (A_ * vec(x) - b).norm() << "\t\t"
              << (vec(C) - vec(Z) - A_.transpose() * y).norm() << "\t\t"
              << (X * Z).norm() << std::endl;
    mu *= 0.0001;
  }
}




void LPSolver2(const Eigen::VectorXd& c, const Eigen::MatrixXd& A,
               const Eigen::VectorXd& b, Eigen::VectorXd& x) {
      FullNTStepIMP(c, A, b, x, OrthantSpace{});
}



void SDPSolver(const Eigen::VectorXd& c, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, Eigen::MatrixXd& X) {
  Eigen::VectorXd x(X.rows() * X.cols());
  std::cout << "FullNTStepIMP Solver" << std::endl;
  SDPIIMP(c, A, b, x);
  std::cout << "FullNTStepIMP Solver Finish" << std::endl;
  X = SemiDefineSpace::Mat(x);
}
Eigen::SparseMatrix<double> ToMatrix(const Eigen::SparseVector<double>& v) {
    size_t n_sqaure = v.rows();
    size_t n = std::sqrt(n_sqaure);
    
    Eigen::SparseMatrix<double> res(n, n);
    const auto* index_ptr = v.innerIndexPtr();
    const double* value_ptr = v.valuePtr();
    size_t nnz = v.nonZeros();
    for (size_t i = 0; i < nnz; ++i) {
        double value = value_ptr[i];
        auto index = index_ptr[i];
        res.insert(index / n, index %n) = value;
    }
    return res;
}

Eigen::SparseMatrix<double> ComputeADAT(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& w) {
    // D = SemiDefineSpace::P(w)
    // (Mat_w * Aj * Mat_w).innerProduct(Ai);
    Eigen::MatrixXd Mat_w = SemiDefineSpace::Mat(w);
    
    std::vector<Eigen::SparseMatrix<double>> Mat_A;
    std::vector<Eigen::MatrixXd> Mat_wAw;
    Mat_A.reserve(A.rows());
    Mat_wAw.reserve(A.rows());
    
    for(size_t i = 0; i < size_t(A.rows()); i++) {
        Eigen::SparseMatrix<double> t = ToMatrix(A.row(i));
        Mat_A.push_back(t);
        Mat_wAw.push_back(Mat_w * t * Mat_w);
    }
    
    Eigen::SparseMatrix<double> res(A.rows(), A.rows());
    using T = Eigen::Triplet<double>;
    std::vector<T> triplet;
    
    for(int col = 0; col < A.rows(); col++) {
        for(int row = 0; row <= col; row++) {
            double value = Mat_A[row].cwiseProduct(Mat_wAw[col]).sum();
            triplet.push_back(T{row, col, value});
        }
    }
    res.setFromTriplets(triplet.begin(), triplet.end());
    return res;
}
Eigen::SparseMatrix<double> ComputeADAT(const std::vector<Eigen::SparseMatrix<double>>& mat_A, const Eigen::VectorXd& w) {
    Eigen::MatrixXd Mat_w = SemiDefineSpace::Mat(w);
    std::vector<Eigen::MatrixXd> Mat_wAw;
    Mat_wAw.reserve(mat_A.size());

    for(size_t i = 0; i < size_t(mat_A.size()); i++) {
        Mat_wAw.push_back(Mat_w * mat_A[i] * Mat_w);
    }

    Eigen::SparseMatrix<double> res(mat_A.size(), mat_A.size());

    using T = Eigen::Triplet<double>;
    std::vector<T> triplet;
    
    for(decltype(mat_A.size()) col = 0; col < mat_A.size(); col++) {
        for(decltype(col) row = 0; row <= col; row++) {
            double value = mat_A[row].cwiseProduct(Mat_wAw[col]).sum();
            triplet.push_back(T{row, col, value});
        }
    }
    res.setFromTriplets(triplet.begin(), triplet.end());
    return res;
}

Eigen::SparseMatrix<double> ComputeADAT(const Eigen::SparseMatrix<double>& A,const Eigen::MatrixXd& D) {
    //Eigen::SparseMatrix<double> AAT = A * Eigen::SparseMatrix<double>(A.transpose());
    Eigen::SparseMatrix<double, Eigen::RowMajor> R_A(A);
    Eigen::MatrixXd DAT = D * A.transpose();
    Eigen::SparseMatrix<double> res(A.rows(), A.rows());
    using T = Eigen::Triplet<double>;
    std::vector<T> triplet;
    
    for(int col = 0; col < A.rows(); col++) {
        for(int row = 0; row <= col; row++) {
            double value = R_A.row(row).dot(DAT.col(col));
            triplet.push_back(T{row, col, value});
        }
    }
    res.setFromTriplets(triplet.begin(), triplet.end());
    
    //std::cout << "Coeffient Error Norm : " << (res - (A * D * A.transpose())).norm() << std::endl;
    return res;
}

auto FeasibleStep(const Eigen::VectorXd& C, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b0,
                  const Eigen::VectorXd& X0, const Eigen::VectorXd& y0, const Eigen::VectorXd& S0,
                  const Eigen::VectorXd& X, const Eigen::VectorXd& S,
                  double delta, double mu0, double theta) {
    using namespace Eigen;
auto start = std::chrono::high_resolution_clock::now();

VectorXd X_sqrt = SemiDefineSpace::Sqrt(X);
VectorXd temp = SemiDefineSpace::Sqrt(SemiDefineSpace::Inverse(SemiDefineSpace::P(X_sqrt, X_sqrt, S)));
VectorXd w = SemiDefineSpace::P(X_sqrt, X_sqrt, temp);
VectorXd sqrt_w = SemiDefineSpace::Sqrt(w);
VectorXd inv_sqrt_w = SemiDefineSpace::Inverse(sqrt_w);
    
std::cout << "Sparse Build w elapse : " << (std::chrono::high_resolution_clock::now() - start).count() / 1000000 << " ms" << std::endl;
//Eigen::MatrixXd Pw_sqrt = SemiDefineSpace::P(SemiDefineSpace::Sqrt(w));
//Eigen::MatrixXd Pw_sqrt_inv = SemiDefineSpace::P(SemiDefineSpace::Sqrt(SemiDefineSpace::Inverse(w)));
//Eigen::MatrixXd Pw = SemiDefineSpace::P(w);
//std::cout << "Sparse Build Pw elapse : " << (std::chrono::high_resolution_clock::now() - start).count() / 1000000 << " ms" << std::endl;
VectorXd rb = b0 - A * X0;
VectorXd rc = C - SparseMatrix<double>(A.transpose()) * y0 - S0;
double mu = delta * mu0;
double inverse_sqrt_mu = 1.0 / (std::sqrt(mu) + std::numeric_limits<double>::epsilon());
//  | a  0   0 |  | dx |   |theta * delta * rb|                                | prim |
//  | 0  b   I |* | dy | = | 1.0 / sqrt(mu) * theta * delta * P(w)^0.5 * rc| = | dual |
//  | I  0   I |  | ds |   | (1 - theta) * v^(-1)  - v|                        | comp |
//   where a = sqrt(mu) * A * P(w)^0.5
//        b = 1.0 / sqrt(mu) * P(w)^0.5 * A^T

    //VectorXd v = inverse_sqrt_mu * Pw_sqrt * S;
    VectorXd v = inverse_sqrt_mu * SemiDefineSpace::P(sqrt_w, sqrt_w, S);
    
    std::cout << "Sparse Delta Distance : " << 0.5 * SemiDefineSpace::Norm(SemiDefineSpace::Inverse(v) - v) << std::endl;
    
    //MatrixXd a = std::sqrt(mu) * A * Pw_sqrt;
    //MatrixXd b = inverse_sqrt_mu * inverse_sqrt_mu * a.transpose(); // Pw_sqrt * A.transpose() and Pw_sqrt is Symmetry

    VectorXd prim = theta * delta * rb;
    //VectorXd dual = inverse_sqrt_mu * theta * delta * Pw_sqrt * rc;
    VectorXd dual = inverse_sqrt_mu * theta * delta * SemiDefineSpace::P(sqrt_w, sqrt_w, rc);
    VectorXd comp = ((1 - theta) * SemiDefineSpace::Inverse(v) - v);

auto end = std::chrono::high_resolution_clock::now();
std::cout << "Sparse Build Problem elapse : " << (end - start).count() / 1000000 << " ms" << std::endl;
start = std::chrono::high_resolution_clock::now();
// Using Schur Complement
//  a * b * dy = prim + a * (dual - comp)
//  since a * b = A * P(w) * A^T
//  prim + a * (dual - comp) = theta * delta * (rb - A * P(w) * rc) - mu * (1 - theta) * A * P(w) * X^(-1) + A * P(w) * S
//  it is better to solve this linear equation with Cholesky Factorization
//
//  dx = b * dy - dual + comp
//   ds = comp - dx
//Vector dy = (a * b).fullPivHouseholderQr().solve(prim + a * (dual - comp));

    //Vector residual = theta *delta * (rb + A * Pw * rc) - mu * (1 - theta) * A * Pw * ConicSpace::Inverse(X) + A * Pw * S;
    //VectorXd dy = (A * Pw * SparseMatrix<double>(A.transpose())).ldlt().solve(prim + a * (dual - comp));
    //Eigen::ConjugateGradient<SparseMatrix<double>, Eigen::Upper> solver;
    SparseMatrix<double> coeffient = ComputeADAT(A, w);
    Eigen::SparseLU<SparseMatrix<double>> solver;
    solver.compute(coeffient.selfadjointView<Upper>());
    //VectorXd dy = solver.solve(prim + a * (dual - comp));
    VectorXd dy = solver.solve(prim + std::sqrt(mu) * (A * SemiDefineSpace::P(sqrt_w, sqrt_w, (dual - comp))));
    //Vector dy = (A * Pw * A.transpose()).ldlt().solve(residual);

    // dx = 1 / (mu) * Pw * A^T * dy - dual + comp
    //VectorXd dx = b * dy - dual + comp;
    VectorXd dx = inverse_sqrt_mu * SemiDefineSpace::P(sqrt_w, sqrt_w, A.transpose() * dy) - dual + comp;
    //std::cout << "dx : " << dx << std::endl;
    VectorXd ds = dual - inverse_sqrt_mu * SemiDefineSpace::P(sqrt_w, sqrt_w, A.transpose() * dy);
    //std::cout << "ds : " << ds << std::endl;
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Sparse Solving elapse : " << (end - start).count() / 1000000  << " ms"<< std::endl;

    std::printf("Sparse Feasible Solution\n");
    std::printf("Sparse delta : %f\n", delta);
    // std::printf("dy solution error = %f\n", (A * Pw *
    // SparseMatrix<double>(A.transpose()) * dy - (prim + a * (dual -
    // comp))).norm());
    std::printf(
        "Norm (a * dx - prim) = %f\n",
        (std::sqrt(mu) * A * SemiDefineSpace::P(sqrt_w, dx, sqrt_w) - prim)
            .norm());
    std::printf("Norm (b * dy + ds - dual) = %f\n",
                (inverse_sqrt_mu *
                     SemiDefineSpace::P(sqrt_w, A.transpose() * dy, sqrt_w) +
                 ds - dual)
                    .norm());
    std::printf("Norm (dx + ds - comp) = %f\n", (dx + ds - comp).norm());
    std::printf("<dx, ds> = %f\n", dx.dot(ds));

    // VectorXd delta_x = std::sqrt(mu) * Pw_sqrt * dx;
    // VectorXd delta_s = std::sqrt(mu) * Pw_sqrt_inv * ds;
    VectorXd delta_x = std::sqrt(mu) * SemiDefineSpace::P(sqrt_w, dx, sqrt_w);
    VectorXd delta_s = std::sqrt(mu) * SemiDefineSpace::P(inv_sqrt_w, ds, inv_sqrt_w);
    
return std::tuple<VectorXd, VectorXd, VectorXd>(delta_x, dy, delta_s);

}

auto CenteringStepImpl(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& X, const Eigen::VectorXd& S, double mu) {
    using namespace Eigen;
  VectorXd X_sqrt = SemiDefineSpace::Sqrt(X);
  VectorXd temp = SemiDefineSpace::Sqrt(SemiDefineSpace::Inverse(SemiDefineSpace::P(X_sqrt, X_sqrt, S)));
  VectorXd w = SemiDefineSpace::P(X_sqrt, X_sqrt, temp);
  Eigen::MatrixXd Pw_sqrt = SemiDefineSpace::P(SemiDefineSpace::Sqrt(w));
  Eigen::MatrixXd Pw_sqrt_inv = SemiDefineSpace::P(SemiDefineSpace::Sqrt(SemiDefineSpace::Inverse(w)));
  Eigen::MatrixXd Pw = SemiDefineSpace::P(w);
  double inverse_sqrt_mu = 1.0 / (std::sqrt(mu));
  VectorXd v = inverse_sqrt_mu * Pw_sqrt * S;
  std::cout << "Delta Distance : " << 0.5 * SemiDefineSpace::Norm(SemiDefineSpace::Inverse(v) - v) << std::endl;
  Eigen::MatrixXd a = std::sqrt(mu) * (A * Pw_sqrt);
  Eigen::MatrixXd b = inverse_sqrt_mu * inverse_sqrt_mu * a.transpose(); // Pw_sqrt * A.transpose() and Pw_sqrt is Symmetry

  VectorXd comp = (SemiDefineSpace::Inverse(v) - v);

  //Vector residual = theta *delta * (rb + A * Pw * rc) - mu * (1 - theta) * A * Pw * ConicSpace::Inverse(X) + A * Pw * S;
  VectorXd dy = (A * Pw * SparseMatrix<double>(A.transpose())).ldlt().solve(a * ( - comp));
  VectorXd dx = b * dy + comp;
  VectorXd ds = -b * dy;

  VectorXd delta_x = std::sqrt(mu) * Pw_sqrt * dx;
  VectorXd delta_s = std::sqrt(mu) * Pw_sqrt_inv * ds;

  return std::tuple<VectorXd, VectorXd, VectorXd>(delta_x, dy, delta_s);
}



void SDPIIMP(const Eigen::VectorXd& C, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, Eigen::VectorXd& X) {
    using namespace Eigen;
    // parameter
    double zeta = 5.0;
    double mu0 = zeta * zeta;
    double epsilon = 1e-7;
    // theta = 1/(4 * r)
    // where r is the rank of Semidefine Matrix of X;
    double theta = 0.25 / std::sqrt(X.rows());
    double delta = 1.0;

    X = SemiDefineSpace::IdentityWithPurterbed(X.rows(), zeta);
    VectorXd S = X;
    Eigen::VectorXd y(A.rows());
    y.setZero();

    VectorXd X0 = X, S0 = S;
    VectorXd y0 = y;
    size_t epoch = 0;


    SparseMatrix<double, RowMajor> R_A(A);
    std::vector<SparseMatrix<double>> mat_A;
    mat_A.reserve(A.rows());
    for(int i = 0; i < A.rows(); i++) {
      mat_A.push_back(ToMatrix(R_A.row(i)));
    }

    VectorXd rb = b - A * X0;
    VectorXd rc = C - A.transpose() * y0 - S0;

    while (Max(SemiDefineSpace::Trace(X, S), (A*X - b).norm(),SemiDefineSpace::Norm(C - A.transpose() * y - S)) > epsilon) {
        // Feasible Step
        auto start = std::chrono::high_resolution_clock::now();
        std::printf("Epoch %zu\t<C, X>=[%.6f]\tPrim Constraint[%.6f]\tDual Constraint[%.6f]\n", ++epoch, C.dot(X), (A*X-b).norm(), SemiDefineSpace::Norm(C-A.transpose() * y - S));

        //   w = P(X^0.5)(P(X^0.5) * S)^-0.5
        VectorXd X_sqrt = SemiDefineSpace::Sqrt(X);
        VectorXd temp = SemiDefineSpace::Sqrt(SemiDefineSpace::Inverse(SemiDefineSpace::P(X_sqrt, X_sqrt, S)));
        VectorXd w = SemiDefineSpace::P(X_sqrt, X_sqrt, temp);
        VectorXd sqrt_w = SemiDefineSpace::Sqrt(w);
        VectorXd inv_sqrt_w = SemiDefineSpace::Inverse(sqrt_w);

        double mu = delta * mu0;
        double inverse_sqrt_mu = 1.0 / (std::sqrt(mu) + std::numeric_limits<double>::epsilon());

        VectorXd v = inverse_sqrt_mu * SemiDefineSpace::P(sqrt_w, S, sqrt_w);
        VectorXd prim = theta * delta * rb;
        VectorXd dual = inverse_sqrt_mu * theta * delta * SemiDefineSpace::P(sqrt_w, sqrt_w, rc);
        VectorXd comp = ((1 - theta) * SemiDefineSpace::Inverse(v) - v);

        SparseMatrix<double> coeffient = ComputeADAT(mat_A, w);
        Eigen::SparseLU<SparseMatrix<double>> solver;
        solver.compute(coeffient.selfadjointView<Upper>());
        VectorXd dy = solver.solve(prim + std::sqrt(mu) * (A * SemiDefineSpace::P(sqrt_w, sqrt_w, (dual - comp))));

        VectorXd dx = inverse_sqrt_mu * SemiDefineSpace::P(sqrt_w, sqrt_w, A.transpose() * dy) - dual + comp;
        VectorXd ds = dual - inverse_sqrt_mu * SemiDefineSpace::P(sqrt_w, sqrt_w, A.transpose() * dy);

        double prim_solution = 
            (std::sqrt(mu) * A * SemiDefineSpace::P(sqrt_w, dx, sqrt_w) - prim).norm();
        double dual_solution =
            (inverse_sqrt_mu * SemiDefineSpace::P(sqrt_w, sqrt_w, A.transpose() * dy) +
             ds - dual)
                .norm();
        double complement_solution = (dx + ds - comp).norm();
        if (Max(prim_solution, dual_solution, complement_solution) > 1e-5) {
          std::printf("Normal Equation Sparse delta [%.6f]\tPrim Constraint [%.6f]\tDual Constraint [%.6f]\tComplement [%.6f]\n", delta, prim_solution, dual_solution, complement_solution);
        }
        

        VectorXd delta_x =
            std::sqrt(mu) * SemiDefineSpace::P(sqrt_w, sqrt_w, dx);
        VectorXd delta_s =
            std::sqrt(mu) * SemiDefineSpace::P(inv_sqrt_w, inv_sqrt_w, ds);
        X += delta_x;
        y += dy;
        S += delta_s;
        delta = (1- theta) * delta;
        mu = delta * mu0;

        if (!SemiDefineSpace::Varify(X)) {
            std::printf("X is infeasible\n");
        }
        
        if (!SemiDefineSpace::Varify(S)) {
            std::printf("S is infeasible\n");
        }
        
        v = ComputeV<SemiDefineSpace>(X, S, mu);
        double delta_distance =  0.5 *SemiDefineSpace::Norm(SemiDefineSpace::Inverse(v) - v); 
        auto end = std::chrono::high_resolution_clock::now();
        if (epoch % 50 == 0) {
          std::cout << "Main Thread Delta Distance : " << delta_distance
                    << std::endl;
          std::cout << "Epoch Elaps : " << (end - start).count() / 1e6 << " ms"
                    << std::endl;
        }

        
        // Centering Path
        /*
        while (delta_distance > 0.5) {
            std::cout << "Delta Distance : " << delta_distance << std::endl; 
            auto [delta_X, delta_y, delta_S] = CenteringStep(A, X, S, mu, ConicSpace{});
            X += delta_X;
            y += delta_y;
            S += delta_S;
            std::cout << "X is feasible : " <<SemiDefineSpace::Varify(X) << std::endl;
            std::cout << "S is feasible : " <<SemiDefineSpace::Varify(S) << std::endl;
            VectorXd v = ComputeV<SemiDefineSpace>(X, S, mu);
            delta_distance =  0.5 *SemiDefineSpace::Norm(SemiDefineSpace::Inverse(v) - v); 
        }
        */
    }
}

Eigen::VectorXd Project(const Eigen::VectorXd& u, int n) {
  Eigen::VectorXd res = u;

  for(int i = 0; i < n; i++) {
    res(i) = u(i) < 0 ? 0 : u(i);
  }
  return res;
}

void PCVI(const Eigen::VectorXd& c, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::VectorXd& x) {
  size_t m = A.rows(), n = A.cols();
  Eigen::MatrixXd M(m + n, m + n);
  M.setZero();
  M.block(0, n, n, m) = -A.transpose();
  M.block(n, 0, m, n) = A;

  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(m + n, m + n);
  Eigen::VectorXd q(m + n);
  q << c, -b; 
  // F(u) = M * u + q
  double beta = 1.0;
  Eigen::VectorXd u(m + n);
  u.setZero();
  double epsilon = 1e-10; 
  int epoch = 0;
  for(epoch = 0;;epoch++) {
    Eigen::VectorXd pu = Project(u - beta * (M * u + q), n);
    double phi = (u - pu).squaredNorm();
    if ( std::sqrt(phi / n) < epsilon) {
      break;
    }
    Eigen::VectorXd direct = (I + beta * M.transpose()) * (u - pu);
    double alpha = phi / direct.squaredNorm();
    beta = sqrt(phi / (M.transpose() * (u - pu)).squaredNorm());
    u = u - alpha * direct;

  }
  std::printf("PCVI %d epochs\n", epoch);
  x = Project(u - beta * (M * u + q), n).block(0, 0, n, 1);

}

void PCVI(const Eigen::VectorXd& c, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, Eigen::VectorXd& x) {
  size_t m = A.rows(), n = A.cols();
  double beta = 1.0;
  double epsilon = 1e-10;
  int epoch = 0;
  double grama = 1.8;
  
  auto F = [A, m, n, c, b](const Eigen::VectorXd& u) {
    Eigen::VectorXd res = u;
    res.block(0, 0, n, 1) = c - A.transpose() * u.block(n, 0, m, 1);
    res.block(n, 0, m, 1) = A * u.block(0, 0, n, 1) - b;
    return res;
  };
  Eigen::VectorXd u (m + n);
  u.setZero();
  for (epoch = 0; ; epoch++) {
    Eigen::VectorXd pu = Project(u - beta * F(u), n);
    double phi = (u - pu).squaredNorm();
    if ( std::sqrt(phi / n) < epsilon) {
      break;
    }
    if (epoch % 100 == 0) {
      std::printf("[%d] error measure = %.13f\n", epoch, std::sqrt(phi / n));
    }
    Eigen::VectorXd direct = u - pu + beta * (F(pu) - F(u));
    double alpha = phi / direct.squaredNorm();
    beta = sqrt(phi / (F(pu) - F(u)).squaredNorm());
    u = u - grama * alpha * direct;
  }
  std::printf("PCVI %d epochs \n", epoch);
  x = Project(u - beta * F(u), n).block(0, 0, n, 1);
}

template<class Matrix, class Vector, class ProjectOperator>
void LVI1(Matrix& M, Vector& q, ProjectOperator&& P, Eigen::VectorXd& u) {
  int n = M.cols();
  u = P(u);
  double beta = 1.0;
  double grama = 1.5;
  double error = std::numeric_limits<double>::infinity();
  int epoch = 0;
  const double EPSILON = 1e-6;
  while(error > EPSILON) {
    Eigen::VectorXd pu = P(u - beta * (M * u + q));
    Eigen::VectorXd direct = beta * (M.transpose() * (u - pu) + M * u + q);
    double phi = (u - pu).squaredNorm();
    double alpha = phi / (u - pu + beta * M.transpose() * (u - pu)).squaredNorm();
    u = P(u - grama * alpha * direct);
    error = std::sqrt(phi / n);
    epoch++;
    if (epoch % 100 == 0) {
      std::printf("[%d] direct measure = %.13f\n", epoch, direct.norm());
    }
  }
}


template<class Matrix, class Vector, class ProjectOperator>
void LVI2(Matrix& M, Vector& q, ProjectOperator&& P, Eigen::VectorXd& u) {
  int n = M.cols();
  double beta = 1.0;
  double grama = 1.5;
  double error = std::numeric_limits<double>::infinity();
  int epoch = 0;
  const double EPSILON = 1e-6;
  while(error > EPSILON) {
    Eigen::VectorXd pu = P(u - beta * (M * u + q));
    Eigen::VectorXd direct = (u - pu) + beta * M.transpose() * (u - pu);
    double phi = (u - pu).squaredNorm();
    double alpha = phi / (direct).squaredNorm();
    u = u - grama * alpha * direct;
    beta = sqrt(phi / (M.transpose() * (u - pu)).squaredNorm());
    error = std::sqrt(phi / n);
    epoch++;
    if (epoch % 100 == 0) {
      std::printf("[%d] error measure = %.13f\n", epoch, error);
    }
  }
}



void PCVI2(const Eigen::VectorXd& c, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, Eigen::VectorXd& x) {
  size_t m = A.rows(), n = A.cols();
  Eigen::SparseMatrix<double> M(n + m, n + m);
  std::vector<Eigen::Triplet<double>> triple;
  triple.reserve(A.nonZeros() * 2);
  for(int k = 0; k < A.outerSize(); k++) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
      int row = it.row();
      int col = it.col();
      double value = it.value();
      triple.emplace_back(n + row, col, value);
      triple.emplace_back(0 + col, n + row, -value);
    }
  }

  M.setFromTriplets(triple.begin(), triple.end());

  Eigen::VectorXd q(n + m);
  q << c, -b;
  Eigen::VectorXd u (m + n);

  auto P = [n](auto&& u) {
    return Project(u, n);
  };
  LVI1(M, q, P, u);
  x = u.block(0, 0, n, 1);
}


Eigen::VectorXd OrthoProject(const Eigen::VectorXd& u) {
  Eigen::VectorXd res = u;
  for(int i = 0; i < res.cols(); i++) {
    if (res(i) < 0) {
      res(i) = 0;
    }
  }
  return res;
}

void ADMM(const Eigen::VectorXd& c, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, Eigen::VectorXd& x) {
  int m = A.rows(), n1 = (A.cols() / 2), n2 = A.cols() - n1;

  Eigen::SparseMatrix<double> A1 = A.leftCols(n1), B1 = A.rightCols(n2);

  Eigen::VectorXd c1 = c.head(n1), c2 = c.tail(n2);

  Eigen::VectorXd lambda(m);
  lambda.setZero();
  Eigen::VectorXd u1(n1), u2(n2);
  u1.setRandom();
  u2.setRandom();
  double beta = 1.0;
  for(int epoch = 0; epoch < 16; epoch++) {
    Eigen::SparseMatrix<double> M1 = beta * Eigen::SparseMatrix<double>(A1.transpose()) * A1;
    Eigen::VectorXd q1 = c1 - A1.transpose() * lambda + beta * A1.transpose() * (B1 * u2 - b);
    LVI1(M1, q1, OrthoProject, u1);

    Eigen::SparseMatrix<double> M2 = beta * Eigen::SparseMatrix<double>(B1.transpose()) * B1;
    Eigen::VectorXd q2 = c2 - B1.transpose() * lambda + beta * B1.transpose() * (A1 * u1 - b);
    LVI1(M1, q1, OrthoProject, u2);
    
    lambda = lambda - beta * (A1 * u1 + B1 * u2 - b);
    std::cout << "ADMM constraint norm : " << (A1 * u1 + B1 * u2 - b).norm() << std::endl;
  }

  x << u1, u2;
}

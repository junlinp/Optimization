#include "linear_programing.h"

#include <assert.h>

#include "iostream"
#include <cmath>

#include "Eigen/SparseCholesky"
#include "Eigen/SparseLU"
#include "Eigen/IterativeLinearSolvers"
namespace internal {
using namespace Eigen;
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> ToSelfEmbeddingProblem(const Eigen::VectorXd& c,const Eigen::MatrixXd& A, const VectorXd& b) {
        MatrixXd::Index m = A.rows();
        MatrixXd::Index n = A.cols();
            
        MatrixXd H_hat(m + n + 1, m + n + 1);
        H_hat.setZero();
        //           |   0     A      -b |
        // H_hat =   |   -A^T  0       c |
        //           |    b^T  -c^T    0 |
        H_hat.block(0, m, m, n) = A;
        H_hat.block(0, m + n, m, 1) = -b;
        
        H_hat.block(m,0, n, m) = -A.transpose();
        H_hat.block(m, m + n, n, 1) = c;
        
        H_hat.block(m + n, 0, 1, n) = b.transpose();
        H_hat.block(m + n, m, 1, n) = -c.transpose();
        
        VectorXd r = VectorXd::Ones(m + n + 1, 1) - H_hat * VectorXd::Ones(m + n + 1, 1);
        MatrixXd H(m + n + 2, m + n + 2);
        H.setZero();
        
        H.block(0, 0 , m + n + 1, m + n + 1) = H_hat;
        H.block(0, m + n + 1, m + n + 1, 1) = r;
        H.block(m + n + 1, 0, 1, m + n + 1) = -r.transpose();
        
        VectorXd q = VectorXd::Zero(m + n + 2, 1);
        q(m + n + 1) = m + n + 2;
        return {H, q};
    }
    MatrixXd SwapPermutation(size_t i, size_t j, size_t n) {
        MatrixXd res = MatrixXd::Identity(n, n);
        res(i, i) = 0;
        res(j, j) = 0;
        res(i, j) = 1;
        res(j, i) = 1;
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
                res = res * SwapPermutation(i, base_index[i], n);
            }
            return res;
        } else {
            // error
            throw std::invalid_argument("Matrix is not full rank");
        }
    }
};
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
void DualLogarithmSolver(const Eigen::VectorXd& c, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::VectorXd& x) {
    size_t m = A.rows(), n = A.cols();
    Eigen::MatrixXd permutation_matrix = internal::Permutation(A);
    Eigen::MatrixXd A_dot = A * permutation;
    Eigen::MatrixXd Ai = A.block(0, 0, m, m);
    Eigen::MatrixXd Aj = A.block(0, m, m, n - m);
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

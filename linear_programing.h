//#define EIGEN_USE_LAPACKE
#define EIGEN_USE_BLAS
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>
#include <iostream>
#include <chrono>

void LPSolver(const Eigen::VectorXd& c, const Eigen::MatrixXd& A,
              const Eigen::VectorXd& b, Eigen::VectorXd& x);
/**
 * @brief solve the LP
 *
 * min c^Tx
 * s.t Ax = b
 *     x >= 0
 **/
void LPSolver2(const Eigen::VectorXd& c, const Eigen::MatrixXd& A,
               const Eigen::VectorXd& b, Eigen::VectorXd& x);
/**
 * @brief solve the robust LP Problem
 *          min <c, x>
 *          s.t A_i * x = bi 
 *              E_i * x <= F_i 
 * @param c 
 * @param A 
 * @param b 
 * @param E 
 * @param F 
 * @param x
 */


void DualLogarithmSolver(const Eigen::VectorXd& c, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::VectorXd& x);

void RobustLPSolver(const Eigen::VectorXd& c,
                    const std::vector<Eigen::VectorXd>& A,
                    const std::vector<double>& b,
                    const std::vector<Eigen::VectorXd>& E,
                    const std::vector<double>& F, Eigen::VectorXd& x);

/**
 *  solve the problem
 *  min tr(C * X)
 *  s.t tr(A_i * X) = b_i for i = 0... m
 *      X >= 0  for X is a semidefinite cone
 */
void SymmetricSolver(const Eigen::MatrixXd C,
                     const std::vector<Eigen::MatrixXd>& A,
                     const Eigen::VectorXd& b, Eigen::MatrixXd& x);

/**
 * @brief solve the SDP problem
 *
 *  min tr(C * X)
 *  s.t tr(A_i * X) = b_i for i = 0... m
 *      X >= 0  for X is a semidefinite cone
 *
 * bibliography:
 *      <Full Newton Step Interior Point Method for Linear Complementarity
 * Problem Over Symmetric Cones> Andrii Berdnikov
 * @param C
 * @param A
 * @param b
 * @param x
 */
void SymmetricSolver2(const Eigen::MatrixXd C,
                      const std::vector<Eigen::MatrixXd>& A,
                      const Eigen::VectorXd& b, Eigen::MatrixXd& x);

void SDPSolver(const Eigen::VectorXd& c, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, Eigen::MatrixXd& X);


class OrthantSpace {
 public:
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

  static Eigen::MatrixXd P(const Vector& v) { return v.array().square().matrix().asDiagonal();};
  static Eigen::MatrixXd L(const Vector& v) { return v.asDiagonal();}
  static Eigen::VectorXd P(const Vector& x, const Vector& y, const Vector& z) {
      return (x.array() * z.array() * y.array()).matrix();
  }

  static Eigen::VectorXd Sqrt(const Vector& v) { return v.array().sqrt().matrix();};

  static Vector Inverse(const Vector& v) { return Vector((1.0 / v.array()).matrix());}
  static double Trace(Vector v) { return v.array().sum();}

  static Vector IdentityWithPurterbed(size_t n, double tau) { Vector v = Vector::Ones(n); v *= tau; return v;}
  static Vector Multiple(const Vector& lhs, const Vector& rhs) {
      return (lhs.array() * rhs.array()).matrix();
  }
  static double Trace(const Vector& lhs, const Vector& rhs) {
      return Multiple(lhs, rhs).array().sum();
  }

  static double Norm(const Vector& v) {
      return std::sqrt(Trace(v, v));
  }
  static bool Varify(const Vector& v) {
      return (v.array() >= 0).all();
  }
};

/*
class LorentzSpace : public ConicOperatorInterface<LorentzSpace> {

  // x mul y = (<x, y>, x0 * y[1:] + y0 * x[1:])
  // x0 >= ||x[1:]||, y0 >= ||y[1:]||
  // x0 * y0 + <x[1:], y[1:]>  >= x0 * y[1:]
  private: 
  Eigen::VectorXd v;
public:
  LorentzSpace(const Eigen::VectorXd& v) : v{v} {};

  LorentzSpace Multiple_Impl(const LorentzSpace& rhs) { 
    Eigen::VectorXd t = v(0) * rhs.v + rhs.v(0) * v;
    t(0) = v.dot(rhs.v);
    return LorentzSpace(t);
  }
  LorentzSpace Sqrt_Impl() const {
      
  }
  LorentzSpace MultipleScalar_Impl(double scalar) const {
      return LorentzSpace(scalar * v);
  }

};
*/

class SemiDefineSpace {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    static Eigen::VectorXd IdentityWithPurterbed(size_t n_n, double zeta) {
        size_t n = std::sqrt(n_n);
        return Vec(Eigen::MatrixXd::Identity(n, n) * zeta);
    }

    
    // It is sparse
    static Eigen::SparseMatrix<double> L(const Eigen::VectorXd& v) {
        size_t n_n = v.rows();
        size_t n = std::sqrt(n_n);
        Eigen::MatrixXd X = Mat(v);
        using T = Eigen::Triplet<double>;

        Eigen::SparseMatrix<double> lhs(n * n, n * n);
        Eigen::SparseMatrix<double> rhs(n * n, n * n);
        std::vector<T> lhs_triple, rhs_triple;
        lhs_triple.reserve(n * n * n);
        rhs_triple.reserve(n * n * n);
        for(size_t col = 0; col < n; col++) {
            for(size_t row = 0; row < n; row++) {
                for(size_t i = 0; i < n; i++) {
                    size_t row_offset = i * n;
                    size_t col_offset = i * n;
                    lhs_triple.push_back(T(row_offset + row, col_offset+col, 0.5 * X(row, col)));
                }
            }
        }
        lhs.setFromTriplets(lhs_triple.begin(), lhs_triple.end());

        for (size_t col = 0; col < n; col++) {
            for(size_t row = 0; row < n; row++) {
                size_t row_offset = row * n;
                size_t col_offset = col * n;
                for(size_t i = 0; i < n; i++) {
                    rhs_triple.push_back(T(row_offset + i, col_offset + i, 0.5 * X(row, col)));
                }
            }
        }
        rhs.setFromTriplets(rhs_triple.begin(), rhs_triple.end());

        return (lhs + rhs);
        
    }

    static Eigen::MatrixXd P(const Eigen::VectorXd& v) {
        size_t n_n = v.rows();
        size_t n = std::sqrt(n_n);
        Eigen::MatrixXd X = Mat(v);
        Eigen::MatrixXd Q(n_n, n_n);
        for(size_t row = 0; row < n; row++) {
            for (size_t col = 0; col < n; col++) {
                Q.block(row * n, col * n, n, n) = X(row, col) * X;
            }
        }
        return Q;
    }

    static Eigen::VectorXd P(const Eigen::VectorXd& X, const Eigen::VectorXd& Y, const Eigen::VectorXd& Z) {
        // 0.5 * (XZY + YZX)

        Eigen::MatrixXd lhs = Mat(X) * Mat(Z) * Mat(Y);
        Eigen::MatrixXd rhs = Mat(Y) * Mat(Z) * Mat(X);

        return 0.5 * Vec(lhs + rhs);
    }

    static Eigen::VectorXd Sqrt(const Eigen::VectorXd& v) {
        Matrix m = Mat(v);
        auto svd = m.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::VectorXd singular_value = svd.singularValues();
        singular_value = singular_value.array().sqrt();
        return Vec(svd.matrixU() * singular_value.asDiagonal() * svd.matrixV().transpose());
    }
    
    static Vector Inverse(const Vector& v) {
        Matrix m = Mat(v);
        auto svd = m.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::VectorXd singular_value = svd.singularValues().array() + std::numeric_limits<double>::epsilon();
        singular_value = singular_value.array().inverse();
        return Vec(svd.matrixU() * singular_value.asDiagonal() * svd.matrixV().transpose());
    }
    
    static Vector Multiple(const Vector& lhs, const Vector& rhs) { 
        return L(lhs) * rhs;
    };

    static double Trace(const Vector& lhs, const Vector& rhs) {
        Vector res = Multiple(lhs, rhs);
        return res.dot(res);
    }

    static double Norm(const Vector& v) {
        return v.norm();
    }

    static Vector Vec(const Matrix& mat) { 
        size_t n = mat.rows(); 
        return Eigen::Map<const Vector>(mat.data(), n * n);
    }

    static Matrix Mat(const Vector& vec) { 
        size_t n_n = vec.rows(); 
        size_t n = std::sqrt(n_n); 
        Matrix r(n, n);
        for(size_t row = 0; row < n; row++) {
            for(size_t col = 0; col < n; col++) {
                r(row, col) = vec(row + col * n);
            }
        }
        return r;
    }

    static bool Varify(const Vector& vec) {
        Eigen::MatrixXd m = Mat(vec);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>  eigen_solver(m);
        if (eigen_solver.info() != Eigen::Success) abort();

        return (eigen_solver.eigenvalues().array().abs() >= 0).all();
    }
};

auto FeasibleStep(const Eigen::VectorXd& C, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b0,
                  const Eigen::VectorXd& X0, const Eigen::VectorXd& y0, const Eigen::VectorXd& S0,
                  const Eigen::VectorXd& X, const Eigen::VectorXd& S,
                  double delta, double mu0, double theta);

template <class Matrix, class Vector>
auto FeasibleStep(const Vector& C, const Matrix& A, const Vector& b0,
                  const Vector& X0, const Vector& y0, const Vector& S0,
                  const Vector& X, const Vector& S,
                  double delta, double mu0, double theta, SemiDefineSpace) {
    return FeasibleStep(C, A, b0, X0, y0, S0, X, S, delta, mu0, theta);
}
                 
template <class Matrix, class Vector, class ConicSpace>
auto FeasibleStep(const Vector& C, const Matrix& A, const Vector& b0,
                  const Vector& X0, const Vector& y0, const Vector& S0,
                  const Vector& X, const Vector& S,
                  double delta, double mu0, double theta, ConicSpace) {
  auto start = std::chrono::high_resolution_clock::now();
  //Eigen::MatrixXd P_X_sqrt = ConicSpace::P(ConicSpace::Sqrt(X));
  Vector X_sqrt = ConicSpace::Sqrt(X);
  //std::cout << "X_sqrt : " << std::endl << X_sqrt << std::endl;

  //Vector w = P_X_sqrt * ConicSpace::Sqrt(ConicSpace::Inverse(P_X_sqrt * S));
  Vector temp = ConicSpace::Sqrt(ConicSpace::Inverse(ConicSpace::P(X_sqrt, X_sqrt, S)));
  //Vector w = P_X_sqrt * ConicSpace::Sqrt(ConicSpace::Inverse(ConicSpace::P(X_sqrt, X_sqrt, S)));
  Vector w = ConicSpace::P(X_sqrt, X_sqrt, temp);
  //std::cout << "w : " << ConicSpace::Varify(w) << std::endl;

  Eigen::MatrixXd Pw_sqrt = ConicSpace::P(ConicSpace::Sqrt(w));
  Eigen::MatrixXd Pw_sqrt_inv = ConicSpace::P(ConicSpace::Sqrt(ConicSpace::Inverse(w)));
  Eigen::MatrixXd Pw = ConicSpace::P(w);

  //std::cout << "P operator" << std::endl << Pw << std::endl;
  //std::cout << "Pw * S : " << std::endl << Pw * S << std::endl;
  Vector rb = b0 - A * X0;
  Vector rc = C - Matrix(A.transpose()) * y0 - S0;
  double mu = delta * mu0;
  double inverse_sqrt_mu = 1.0 / (std::sqrt(mu) + std::numeric_limits<double>::epsilon());
  //  | a  0   0 |  | dx |   |theta * delta * rb|                                | prim | 
  //  | 0  b   I |* | dy | = | 1.0 / sqrt(mu) * theta * delta * P(w)^0.5 * rc| = | dual |
  //  | I  0   I |  | ds |   | (1 - theta) * v^(-1)  - v|                        | comp |
  //   where a = sqrt(mu) * A * P(w)^0.5
  //        b = 1.0 / sqrt(mu) * P(w)^0.5 * A^T


  Vector v = inverse_sqrt_mu * Pw_sqrt * S;
  std::cout << "Delta Distance : " << 0.5 * ConicSpace::Norm(ConicSpace::Inverse(v) - v) << std::endl; 
  Matrix a = std::sqrt(mu) * A * Pw_sqrt;
  Matrix b = inverse_sqrt_mu * inverse_sqrt_mu * a.transpose(); // Pw_sqrt * A.transpose() and Pw_sqrt is Symmetry

  Vector prim = theta * delta * rb;
  Vector dual = inverse_sqrt_mu * theta * delta * Pw_sqrt * rc;
  Vector comp = ((1 - theta) * ConicSpace::Inverse(v) - v);
  //std::cout << "Prim : " << prim << std::endl;
  //std::cout << "dual : " << dual << std::endl;
  //std::cout << "comp : " << comp << std::endl;
  //std::cout << "a : " << a << std::endl;
  //std::cout << "b : " << b << std::endl;
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Build Problem elapse : " << (end - start).count() / 1000000 << " ms" << std::endl;
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

  Vector residual = theta *delta * (rb + A * Pw * rc) - mu * (1 - theta) * A * Pw * ConicSpace::Inverse(X) + A * Pw * S;
  Vector dy = (A * Pw * Matrix(A.transpose())).ldlt().solve(prim + a * (dual - comp));
  //Vector dy = (A * Pw * A.transpose()).ldlt().solve(residual);
  //std::cout << "dy : " << dy << std::endl;
  Vector dx = b * dy - dual + comp;
  //std::cout << "dx : " << dx << std::endl;
  Vector ds = dual - b * dy;
  //std::cout << "ds : " << ds << std::endl;
  end = std::chrono::high_resolution_clock::now();
  std::cout << "Solving elapse : " << (end - start).count() / 1000000  << " ms"<< std::endl;

  std::printf("Feasible Solution\n");
  std::printf("delta : %f\n", delta);
  std::printf("dy solution error = %f\n", (A * Pw * Matrix(A.transpose()) * dy - (prim + a * (dual - comp))).norm());
  std::printf("Norm (a * dx - prim) = %f\n", (a * dx - prim).norm());
  std::printf("Norm (b * dy + ds - dual) = %f\n", (b * dy + ds - dual).norm());
  std::printf("Norm (dx + ds - comp) = %f\n", (dx + ds - comp).norm());
  std::printf("<dx, ds> = %f\n", dx.dot(ds));


  Vector delta_x = std::sqrt(mu) * Pw_sqrt * dx;
  Vector delta_s = std::sqrt(mu) * Pw_sqrt_inv * ds;

  return std::tuple<Vector, Vector, Vector>(delta_x, dy, delta_s);
}


auto CenteringStepImpl(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& X, const Eigen::VectorXd& S, double mu);


template <class Matrix, class Vector>
auto CenteringStep(const Matrix& A, const Vector& X, const Vector&S, double mu, SemiDefineSpace ) {
    return CenteringStepImpl(A, X, S, mu);
}

template <class ConicSpace, class Matrix, class Vector>
auto CenteringStep(const Matrix& A, const Vector& X, const Vector&S, double mu, ConicSpace) {

  Vector X_sqrt = ConicSpace::Sqrt(X);
  Vector temp = ConicSpace::Sqrt(ConicSpace::Inverse(ConicSpace::P(X_sqrt, X_sqrt, S)));
  Vector w = ConicSpace::P(X_sqrt, X_sqrt, temp);
  Eigen::MatrixXd Pw_sqrt = ConicSpace::P(ConicSpace::Sqrt(w));
  Eigen::MatrixXd Pw_sqrt_inv = ConicSpace::P(ConicSpace::Sqrt(ConicSpace::Inverse(w)));
  Eigen::MatrixXd Pw = ConicSpace::P(w);
  double inverse_sqrt_mu = 1.0 / (std::sqrt(mu));
  Vector v = inverse_sqrt_mu * Pw_sqrt * S;
  std::cout << "Delta Distance : " << 0.5 * ConicSpace::Norm(ConicSpace::Inverse(v) - v) << std::endl; 
  Eigen::MatrixXd a = std::sqrt(mu) * (A * Pw_sqrt);
  Eigen::MatrixXd b = inverse_sqrt_mu * inverse_sqrt_mu * a.transpose(); // Pw_sqrt * A.transpose() and Pw_sqrt is Symmetry

  Vector comp = (ConicSpace::Inverse(v) - v);

  //Vector residual = theta *delta * (rb + A * Pw * rc) - mu * (1 - theta) * A * Pw * ConicSpace::Inverse(X) + A * Pw * S;
  Vector dy = (A * Pw * Matrix(A.transpose())).ldlt().solve(a * ( - comp));
  Vector dx = b * dy + comp;
  Vector ds = b * dy;

  Vector delta_x = std::sqrt(mu) * Pw_sqrt * dx;
  Vector delta_s = std::sqrt(mu) * Pw_sqrt_inv * ds;

  return std::tuple<Vector, Vector, Vector>(delta_x, dy, delta_s);
}
inline double Max(double a, double b, double c) {
    return std::max(c, std::max(a, b));
}

template<class ConicSpace>
Eigen::VectorXd ComputeV(const Eigen::VectorXd& X,const Eigen::VectorXd& S,double mu) {
  Eigen::VectorXd X_sqrt = ConicSpace::Sqrt(X);
  Eigen::VectorXd temp = ConicSpace::Sqrt(ConicSpace::Inverse(ConicSpace::P(X_sqrt, X_sqrt, S)));
  Eigen::VectorXd w = ConicSpace::P(X_sqrt, X_sqrt, temp);
  Eigen::VectorXd w_sqrt = ConicSpace::Sqrt(w);
  double inverse_sqrt_mu = 1.0 / (std::sqrt(mu) + std::numeric_limits<double>::epsilon());
  //return inverse_sqrt_mu * Pw_sqrt * S;
  return inverse_sqrt_mu * ConicSpace::P(w_sqrt, S, w_sqrt);
}

template<class Matrix, class Vector, class ConicSpace>
void FullNTStepIMP(const Vector& C,const Matrix& A, const Vector& b,Vector& X, ConicSpace) {
    //
    // min <C, X>
    // s.t <Ai, X> = bi
    //      X belongs to the Conic Space such as Orthant, Ice cream cone, Semi Definite cone
    //
    
    //
    // Initialize the variable
    //
    // Warning: 
    // there must be let X^(*) + S^(*) <=_(K) zeta * e
    // 
    double zeta = 5.0;
    //
    double mu0 = zeta * zeta;
    double epsilon = 1e-7;

    double theta = 1.0 / std::sqrt(2 * X.rows());
    double delta = 1.0;


    X = ConicSpace::IdentityWithPurterbed(X.rows(), zeta);
    Vector S = X;
    Eigen::VectorXd y(A.rows());
    y.setZero();

    Vector X0 = X, S0 = S;
    Eigen::VectorXd y0 = y;
    size_t epoch = 0;
    std::cout << "Here" << std::endl;
    //std::cout << "L(X) : " << ConicSpace::L(X) << std::endl;
    std::cout << "Trace(X, S) : " << ConicSpace::Trace(X, S) << std::endl;
    std::cout << "Primal Constraint Norm : " << (A * X - b).norm() << std::endl;
    std::cout << "Dual constraint Norm : " << ConicSpace::Norm(C - A.transpose() * y - S) << std::endl;
    while (Max(ConicSpace::Trace(X, S), (A*X - b).norm(), ConicSpace::Norm(C - A.transpose() * y - S)) > epsilon) {
        // Feasible Step
        std::cout << "Epoch : " <<  ++epoch << std::endl;
        std::cout << "Trace(X, S) : " << ConicSpace::Trace(X, S) << std::endl;
        std::cout << "Primal Constraint Norm : " << (A * X - b).norm() << std::endl;
        std::cout << "Dual constraint Norm : " << ConicSpace::Norm(C - A.transpose() * y - S) << std::endl;

        auto [delta_X, delta_y, delta_S] = FeasibleStep(C, A, b, X0, y0, S0, X, S, delta, mu0, theta, ConicSpace{});
        X += delta_X;
        y += delta_y;
        S += delta_S;

        delta = (1- theta) * delta;
        double mu = delta * mu0;
        std::cout << "X is feasible : " << ConicSpace::Varify(X) << std::endl;
        std::cout << "S is feasible : " << ConicSpace::Varify(S) << std::endl;

        Vector v = ComputeV<ConicSpace>(X, S, mu);
        double delta_distance =  0.5 * ConicSpace::Norm(ConicSpace::Inverse(v) - v); 
        std::cout << "Main Thread Delta Distance : " << delta_distance << std::endl; 
        
        // Centering Path
        while (delta_distance > 0.5) {
            std::cout << "Delta Distance : " << delta_distance << std::endl; 
            auto [delta_X, delta_y, delta_S] = CenteringStep(A, X, S, mu, ConicSpace{});
            X += delta_X;
            y += delta_y;
            S += delta_S;
            std::cout << "X is feasible : " << ConicSpace::Varify(X) << std::endl;
            std::cout << "S is feasible : " << ConicSpace::Varify(S) << std::endl;
            Vector v = ComputeV<ConicSpace>(X, S, mu);
            delta_distance =  0.5 * ConicSpace::Norm(ConicSpace::Inverse(v) - v); 
        }
        //std::cout << "y : " << y << std::endl;
   
        //std::printf("Norm of Primal Constraint %f\n", (A * X - b).norm());
        //std::printf("Norm of Dual Constraint %f\n", ConicSpace::Norm(C - A.transpose() * y - S));
        //std::printf("Gap : %f\n", ConicSpace::Trace(X, S));
        
    }

    //std::printf("Norm of Primal Constraint %f\n", Norm(A * X.ToLinearVector() - b));
    //std::printf("Norm of Dual Constraint %f\n", Norm(C - A.transpose() * y - S.ToLinearVector()));
}

void SDPIIMP(const Eigen::VectorXd& C, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, Eigen::VectorXd& X);

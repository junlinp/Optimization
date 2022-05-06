#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/KroneckerProduct>

#include <vector>
#include <iostream>

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

void SDPSolver(const Eigen::VectorXd& c, const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::MatrixXd& X);


class OrthantSpace {
 public:
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

  static Eigen::MatrixXd P(const Vector& v) { return v.array().square().matrix().asDiagonal();};
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
    static Eigen::MatrixXd L(const Eigen::VectorXd& v) {
        size_t n_n = v.rows();
        size_t n = std::sqrt(n_n);
        return 0.5 * (Eigen::kroneckerProduct(Eigen::MatrixXd::Identity(n, n), Mat(v)) + Eigen::kroneckerProduct(Mat(v), Eigen::MatrixXd::Identity(n, n)));
    }

    static Eigen::MatrixXd P(const Eigen::VectorXd& v) {
        return 2 * L(v) * L(v) - L(Multiple(v, v));
    }

    static Eigen::VectorXd Sqrt(const Eigen::VectorXd& v) {
        Matrix m = Mat(v);
        auto svd = m.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::VectorXd singular_value = svd.singularValues();
        singular_value.array().sqrt();
        return Vec(svd.matrixU() * singular_value.asDiagonal() * svd.matrixV().transpose());
    }
    static Vector Inverse(const Vector& v) {
        Matrix m = Mat(v);
        auto svd = m.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::VectorXd singular_value = svd.singularValues();
        singular_value.array().inverse();
        return Vec(svd.matrixU() * singular_value.asDiagonal() * svd.matrixV().transpose());
    }
    static Vector Multiple(const Vector& lhs, const Vector& rhs) { 
        Matrix l = Mat(lhs), r = Mat(rhs); 
        return Vec(((lhs * rhs) + (rhs * lhs)) * 0.5);
        };

    static double Trace(const Vector& lhs, const Vector& rhs) {
        return (Mat(lhs).transpose() * Mat(rhs)).trace();
    }

    static double Norm(const Vector& v) {
        return Mat(v).norm();
    }

    static Vector Vec(const Matrix& mat) { 
        size_t n = mat.rows(); 
        return Eigen::Map<const Vector>(mat.data(), n * n);
    }

    static Matrix Mat(const Vector& vec) { 
        size_t n_n = vec.rows(); 
        size_t n = std::sqrt(n); 
        return Eigen::Map<const Matrix>(vec.data(), n, n);
        }
};

template <class Matrix, class Vector, class ConicSpace>
auto FeasibleStep(const Vector& C, const Matrix& A, const Vector& b0,
                  const Vector& X0, const Vector& y0, const Vector& S0,
                  const Vector& X, const Vector& y, const Vector& S,
                  double delta, double mu0, double theta, ConicSpace) {
  
  Matrix P_X_sqrt = ConicSpace::P(ConicSpace::Sqrt(X));
  Vector w = P_X_sqrt * ConicSpace::Sqrt(ConicSpace::Inverse(P_X_sqrt * S));

  std::cout << "X : " << X << std::endl;
  std::cout << "S : " << S << std::endl;
  std::cout << "w : " << w << std::endl;

  Matrix Pw_sqrt = ConicSpace::P(ConicSpace::Sqrt(w));
  Matrix Pw_sqrt_inv = ConicSpace::P(ConicSpace::Sqrt(ConicSpace::Inverse(w)));

  Vector rb = b0 - A * X0;
  Vector rc = C - A.transpose() * y0 - S0;
  
  double mu = delta * mu0;
  double inverse_sqrt_mu = 1.0 / (std::sqrt(mu) + std::numeric_limits<double>::epsilon());
  //  | a  0   0 |  | dx |   |theta * delta * rb|                                | prim | 
  //  | 0  b   I |* | dy | = | 1.0 / sqrt(mu) * theta * delta * P(w)^0.5 * rc| = | dual |
  //  | I  0   I |  | ds |   | (1 - theta) * v^(-1)  - v|                        | comp |
  //   where a = sqrt(mu) * A * P(w)^0.5
  //        b = 1.0 / sqrt(mu) * P(w)^0.5 * A^T


  Vector v = inverse_sqrt_mu * Pw_sqrt_inv * X;

  Matrix a = std::sqrt(mu) * A * Pw_sqrt;
  Matrix b = inverse_sqrt_mu * Pw_sqrt * A.transpose();

  Vector prim = theta * delta * rb;
  Vector dual = inverse_sqrt_mu * theta * delta * Pw_sqrt * rc;
  Vector comp = ((1 - theta) * ConicSpace::Inverse(v) - v);

  // Using Schur Complement
  //  a * b * dy = prim + a * (dual - comp)
  //  since a * b = A * P(w) * A^T
  //  prim + a * (dual - comp) = theta * delta * (rb - A * P(w) * rc) - mu * (1 - theta) * A * P(w) * X^(-1) + A * P(w) * S
  //  it is better to solve this linear equation with Cholesky Factorization
  //
  //  dx = b * dy - dual + comp
  //   ds = comp - dx
  //Vector dy = (a * b).fullPivHouseholderQr().solve(prim + a * (dual - comp));
  Matrix Pw = ConicSpace::P(w);
  Vector residual = theta *delta * (rb + A * Pw * rc) - mu * (1 - theta) * A * Pw * ConicSpace::Inverse(X) + A * Pw * S;
  //Vector dy = (A * w.P() * A.transpose()).ldlt().solve(prim + a * (dual - comp));
  Vector dy = (A * Pw * A.transpose()).ldlt().solve(residual);

  Vector dx = b * dy - dual + comp;
  Vector ds = dual - b * dy;
/*
  std::printf("Feasible Solution\n");
  std::printf("delta : %f\n", delta);
  std::printf("Norm (a * dx - prim) = %f\n", Norm(a * dx - prim));
  std::printf("Norm (b * dy + ds - dual) = %f\n", Norm(b * dy + ds - dual));
  std::printf("Norm (dx + ds) = %f\n", Norm(comp));
  std::printf("<dx, ds> = %f\n", dx.dot(ds));
  */

  Vector delta_x = std::sqrt(mu) * Pw_sqrt * dx;
  Vector delta_s = std::sqrt(mu) * Pw_sqrt_inv * ds;

  return std::tuple<Vector, Vector, Vector>(delta_x, dy, delta_s);
}
inline double Max(double a, double b, double c) {
    return std::max(c, std::max(a, b));
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
    double zeta = 1024.0;
    //
    double mu0 = zeta * zeta;
    double epsilon = 1e-8;

    double tau = 0.25;
    double theta = 0.5;
    double delta = 1.0;


    X = ConicSpace::IdentityWithPurterbed(X.rows(), zeta);
    Vector S = X;
    Eigen::VectorXd y(A.rows());
    y.setZero();

    Vector X0 = X, S0 = S;
    Eigen::VectorXd y0 = y;

    while (Max(ConicSpace::Trace(X, S), (A*X - b).norm(), ConicSpace::Norm(C - A.transpose() * y - S)) > epsilon) {
        // Feasible Step
        auto [delta_X, delta_y, delta_S] = FeasibleStep(C, A, b, X0, y0, S0, X, y, S, delta, mu0, theta, ConicSpace{});
        X += delta_X;
        y += delta_y;
        S += delta_S;

        delta = (1- theta) * delta;
        std::cout << "X : " << X << std::endl;
        std::cout << "y : " << y << std::endl;
        std::cout << "S : " << S << std::endl;
   
        //std::printf("Norm of Primal Constraint %f\n", Norm(A * X.ToLinearVector() - b));
        //std::printf("Norm of Dual Constraint %f\n", Norm(C - A.transpose() * y - S.ToLinearVector()));
        //std::printf("Gap : %f\n", Trace(X, S));
        
    }

    //std::printf("Norm of Primal Constraint %f\n", Norm(A * X.ToLinearVector() - b));
    //std::printf("Norm of Dual Constraint %f\n", Norm(C - A.transpose() * y - S.ToLinearVector()));
}
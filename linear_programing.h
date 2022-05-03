#include <Eigen/Dense>
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

template <class Derive>
class ConicOperatorInterface {
 public:
  // Quadratic Linear Operator
  Eigen::MatrixXd P() { return static_cast<Derive*>(this)->P_Impl(); }

  Derive Sqrt() const { return static_cast<const Derive*>(this)->Sqrt_Impl(); }

  Derive MultipleScalar(double scalar) { return static_cast<Derive*>(const_cast<Derive*>(this))->MultipleScalar_Impl(scalar);}
  Derive MultipleScalar(double scalar) const { return static_cast<const Derive*>(this)->MultipleScalar_Impl(scalar);}
  Eigen::VectorXd ToLinearVector() { return static_cast<Derive*>(const_cast<ConicOperatorInterface*>(this))->ToLinearVector_Impl();}
  Eigen::VectorXd ToLinearVector() const { return static_cast<const Derive*>(this)->ToLinearVector_Impl();}

  Derive Inverse() const { return static_cast<const Derive*>(this)->Inverse_Impl();}

  void IdentityWithPurterbed(double tau) { return static_cast<Derive*>(this)->IdentityWithPurterbed_Impl(tau);}
};

class OrthantSpace : public ConicOperatorInterface<OrthantSpace> {
 private:
  Eigen::VectorXd v;

 public:
  explicit OrthantSpace (Eigen::VectorXd x) : v{x} {};

  Eigen::MatrixXd P_Impl() const { return v.array().square().matrix().asDiagonal();};
  OrthantSpace Sqrt_Impl() const { return OrthantSpace(v.array().sqrt().matrix());};
  OrthantSpace Inverse_Impl() const { return OrthantSpace((1.0 / v.array()).matrix());}
  OrthantSpace MultipleScalar_Impl(double scalar) const { return OrthantSpace(scalar * v);}
  OrthantSpace Multiple(const OrthantSpace& rhs) const { return OrthantSpace((v.array() * rhs.v.array()).matrix());}
  double Trace() const { return v.array().sum();}

  void IdentityWithPurterbed_Impl(double tau) { v.setOnes(); v *= tau;}

  size_t size() const { return v.rows();}
  OrthantSpace LMultipleMatrix(const Eigen::MatrixXd& lhs) const { return OrthantSpace(lhs * v);}

  Eigen::VectorXd ToLinearVector_Impl() const { return v;}

  OrthantSpace operator+(const OrthantSpace& rhs) const{
      return OrthantSpace(v + rhs.v);
  }
  OrthantSpace operator-(const OrthantSpace& rhs) const{
      return OrthantSpace(v - rhs.v);
  }

  OrthantSpace& operator+=(const OrthantSpace& rhs) {
      v += rhs.v;
      v = (v.array() + std::numeric_limits<double>::epsilon()).matrix();
      return *this;
  }
};

inline OrthantSpace operator*(double lhs, const OrthantSpace& rhs) {
    return rhs.MultipleScalar(lhs);
}

inline OrthantSpace operator*(const OrthantSpace& lhs, double rhs) {
    return lhs.MultipleScalar(rhs);
}

inline double Trace(const OrthantSpace& lhs, const OrthantSpace& rhs) {
    return lhs.Multiple(rhs).Trace();
}
inline double Norm(Eigen::VectorXd x) {
    return x.norm();
}

inline OrthantSpace operator*(const Eigen::MatrixXd& lhs, const OrthantSpace& rhs) {
    return rhs.LMultipleMatrix(lhs);
}
template <class Matrix, class Vector, class ConicSpace>
auto FeasibleStep(const Vector& C, const Matrix& A, const Vector& b0,
                  const ConicSpace& X0, const Vector& y0, const ConicSpace& S0,
                  const ConicSpace& X, const Vector& y, const ConicSpace& S,
                  double delta, double mu0, double theta) {

  Matrix P_X_sqrt = X.Sqrt().P();
  ConicSpace w(Eigen::VectorXd(
      (P_X_sqrt * (ConicSpace(P_X_sqrt * S.ToLinearVector()).Sqrt().Inverse())
                      .ToLinearVector())
          .eval()));

  std::cout << "X : " << X.ToLinearVector() << std::endl;
  std::cout << "S : " << S.ToLinearVector() << std::endl;
  std::cout << "w : " << w.ToLinearVector() << std::endl;

  Matrix Pw_sqrt = w.Sqrt().P();
  Matrix Pw_sqrt_inv = w.Inverse().Sqrt().P();

  Vector rb = b0 - A * X0.ToLinearVector();
  Vector rc = C - A.transpose() * y0 - S0.ToLinearVector();
  
  double mu = delta * mu0;
  double inverse_sqrt_mu = 1.0 / (std::sqrt(mu) + std::numeric_limits<double>::epsilon());
  //  | a  0   0 |  | dx |   |theta * delta * rb|                                | prim | 
  //  | 0  b   I |* | dy | = | 1.0 / sqrt(mu) * theta * delta * P(w)^0.5 * rc| = | dual |
  //  | I  0   I |  | ds |   | (1 - theta) * v^(-1)  - v|                        | comp |
  //   where a = sqrt(mu) * A * P(w)^0.5
  //        b = 1.0 / sqrt(mu) * P(w)^0.5 * A^T


  ConicSpace v = inverse_sqrt_mu * Pw_sqrt_inv * X;

  Matrix a = std::sqrt(mu) * A * Pw_sqrt;
  Matrix b = inverse_sqrt_mu * Pw_sqrt * A.transpose();

  Vector prim = theta * delta * rb;
  Vector dual = inverse_sqrt_mu * theta * delta * Pw_sqrt * rc;
  Vector comp = ((1 - theta) * v.Inverse() - v).ToLinearVector();

  // Using Schur Complement
  //  a * b * dy = prim + a * (dual - comp)
  //  dx = b * dy - dual + comp
  //   ds = comp - dx

  Vector dy = (a * b).fullPivHouseholderQr().solve(prim + a * (dual - comp));
  Vector dx = b * dy - dual + comp;
  Vector ds = dual - b * dy;

  std::printf("Feasible Solution\n");
  std::printf("delta : %f\n", delta);
  std::printf("Norm (a * dx - prim) = %f\n", Norm(a * dx - prim));
  std::printf("Norm (b * dy + ds - dual) = %f\n", Norm(b * dy + ds - dual));
  std::printf("Norm (dx + ds) = %f\n", Norm(comp));

  Vector delta_x = std::sqrt(mu) * Pw_sqrt * dx;
  Vector delta_s = std::sqrt(mu) * Pw_sqrt_inv * ds;

  return std::tuple<ConicSpace, Vector, ConicSpace>(delta_x, dy, delta_s);
}
inline double Max(double a, double b, double c) {
    return std::max(c, std::max(a, b));
}
template<class Matrix, class Vector, class ConicSpace>
void FullNTStepIMP(const Vector& C,const Matrix& A, const Vector& b, ConicSpace& X) {
    //
    // min <C, X>
    // s.t <Ai, X> = bi
    //      X belongs to the Conic Space such as Orthant, Ice cream cone, Semi Definite cone
    //
    
    //
    // Initialize the variable
    //
    double zeta = 1.0;
    double mu0 = zeta * zeta;
    double epsilon = 1e-6;

    double tau = 0.25;
    double theta = 0.5;
    double delta = 1.0;


    X.IdentityWithPurterbed(zeta);
    ConicSpace S = X;
    Eigen::VectorXd y(A.rows());
    y.setZero();

    ConicSpace X0 = X, S0 = S;
    Eigen::VectorXd y0 = y;

    while (Max(Trace(X, S), Norm(A*X.ToLinearVector() - b), Norm(C - A.transpose() * y - S.ToLinearVector())) > epsilon) {
        // Feasible Step
        auto [delta_X, delta_y, delta_S] = FeasibleStep(C, A, b, X0, y0, S0, X, y, S, delta, mu0, theta);
        X += delta_X;
        y += delta_y;
        S += delta_S;

        delta = (1- theta) * delta;
        std::cout << "X : " << X.ToLinearVector() << std::endl;
        std::cout << "y : " << y << std::endl;
        std::cout << "S : " << S.ToLinearVector() << std::endl;

        std::printf("Norm of Primal Constraint %f\n", Norm(A * X.ToLinearVector() - b));
        std::printf("Norm of Dual Constraint %f\n", Norm(C - A.transpose() * y - S.ToLinearVector()));
        
    }

    std::printf("Norm of Primal Constraint %f\n", Norm(A * X.ToLinearVector() - b));
    std::printf("Norm of Dual Constraint %f\n", Norm(C - A.transpose() * y - S.ToLinearVector()));
}
#ifndef OPTIMIZATION_LINEAR_SOLVER_H_
#define OPTIMIZATION_LINEAR_SOLVER_H_
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "iostream"
class AbstratorCoefficient {
 public:
  /*
   * A * rhs
   *
   */
  virtual Eigen::VectorXd Multiple(const Eigen::VectorXd& rhs) = 0;

  /*
   *  dot(A * rhs, rhs)
   */
  virtual double MultipleDot(const Eigen::VectorXd& rhs) = 0;

  virtual size_t Rows() = 0;

  virtual size_t Cols() = 0;
};

/*
 *
 * This is the Coefficient (A^T * A) for a Normal formla such as
 * A^T * A * x = b
 *
 */
template <class T>
class NormalFormulaCoefficient : public AbstratorCoefficient {
 public:
  NormalFormulaCoefficient(const Eigen::SparseMatrix<T>& A) : A_(A) {}
  virtual ~NormalFormulaCoefficient() {}
  Eigen::VectorXd Multiple(const Eigen::VectorXd& rhs) override {
    Eigen::VectorXd temp = A_ * rhs;
    return A_.transpose() * temp;
  }

  /*
   * dot(A^T * A * rhs, rhs) = dot(A * rhs, A * rhs)
   */
  double MultipleDot(const Eigen::VectorXd& rhs) override {
    Eigen::VectorXd temp = A_ * rhs;
    return temp.dot(temp);
  }

  size_t Rows() override {
      return A_.cols();
  }

  size_t Cols() override {
      return A_.cols();
  }

 private:
  Eigen::SparseMatrix<T> A_;
};

/*
 * (J^T * J + lambda * D^T D) x = -J^T f
 *
 */
template <class T>
class BundleAdjustmentNormalFormulaCoefficient : public AbstratorCoefficient {
 public:
  BundleAdjustmentNormalFormulaCoefficient(
      const Eigen::SparseMatrix<T>& jacobian,
      const Eigen::SparseMatrix<T>& diagonal, double lambda)
      : jacobian_(jacobian), diagonal_(diagonal), lambda_(lambda) {}
  virtual ~BundleAdjustmentNormalFormulaCoefficient() {}

  Eigen::VectorXd Multiple(const Eigen::VectorXd& rhs) override {
    Eigen::VectorXd temp1 = jacobian_.Multiple(rhs);
    Eigen::VectorXd temp2 = diagonal_.Multiple(rhs);
    return temp1 + lambda_ * temp2;
  }
  
  
  double MultipleDot(const Eigen::VectorXd& rhs) override {
    double jacobian_dot_rhs = jacobian_.MultipleDot(rhs);
    double diagonal_dot_rhs = diagonal_.MultipleDot(rhs);
    return jacobian_dot_rhs + lambda_ * diagonal_dot_rhs;
  }
  size_t Rows() override { return jacobian_.Rows(); }

  size_t Cols() override { return jacobian_.Cols(); }

 private:
  NormalFormulaCoefficient<T> jacobian_;
  NormalFormulaCoefficient<T> diagonal_;
  double lambda_;
};

template <class Matrix, class Vector>
void ConjugateGradient(const Matrix& A, const Vector& b, Vector& x) {
  Vector r = b - A * x;
  Vector p = r;

  int max_iterator = 8196;
  int iterator = 0;
  double last_r_dot_r = r.dot(r);
  while (iterator++ < max_iterator) {
    if (r.norm() < 1e-5 * r.rows()) {
      return;
    }
    Vector temp = A * p;
    double alpha = last_r_dot_r / p.dot(temp);
    x = x + alpha * p;
    r = r - alpha * temp;
    double new_r_dot_r = r.dot(r);
    double beta = new_r_dot_r / last_r_dot_r;
    p = r + beta * p;
    last_r_dot_r = new_r_dot_r;
  }
}

void ConjugateGradient(std::shared_ptr<AbstratorCoefficient> A,
                       const Eigen::VectorXd& b, Eigen::VectorXd& x) {
  size_t max_iterator = A->Cols();
  size_t iterator = 0;
  x = Eigen::VectorXd::Zero(b.rows());
  Eigen::VectorXd r = b - A->Multiple(x);
  Eigen::VectorXd p = r;
  while(iterator++ < max_iterator) {
      if (r.norm() < 1e-5) {
          return;
      }
      double r_dot_r = r.dot(r);
      double alpha = r_dot_r / A->MultipleDot(p);
      x += alpha * p;
      r -= alpha * A->Multiple(p);
      double beta = r.dot(r) / r_dot_r;
      p = r + beta * p;
  }
}

#endif  // OPTIMIZATION_LINEAR_SOLVER_H_
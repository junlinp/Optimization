#ifndef SOLVER_LM_H_
#define SOLVER_LM_H_
#include "Eigen/Dense"
#include <memory>

class FirstOrderFunction {
 public:
  virtual size_t VariableDimension() const = 0;
  virtual Eigen::MatrixXd Jacobians(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::VectorXd Evaluate(const Eigen::VectorXd& x) const = 0;
};

class BinaryFunction : public FirstOrderFunction {
 public:
  // virtual size_t VariableDimension() const = 0;
  // virtual Eigen::MatrixXd Jacobians(const Eigen::VectorXd& x) const = 0;
  // virtual Eigen::VectorXd Evaluate(const Eigen::VectorXd& x) const = 0;

  virtual void BinaryJacobians(
      Eigen::MatrixXd* first_part_jacobians,
      Eigen::MatrixXd* second_part_jacobians) const = 0;
};

class LMSolver {
 public:
  LMSolver() = default;
  virtual ~LMSolver() = default;

  void SetFunction(std::shared_ptr<FirstOrderFunction> function) {
    function_ = function;
  }
  bool Solve(Eigen::VectorXd* x);
  bool AcceptStepOrNot(const Eigen::VectorXd& x, const Eigen::VectorXd& step,
                       double* trust_region);

 private:
  std::shared_ptr<FirstOrderFunction> function_ = nullptr;
};

class BundleAdjustmentLMSolver {
 public:
  BundleAdjustmentLMSolver() = default;
  virtual ~BundleAdjustmentLMSolver() = default;

  void SetFunction(std::shared_ptr<BinaryFunction> function) {
    function_ = function;
  }
  bool Solve(Eigen::VectorXd* x);
  bool AcceptStepOrNot(const Eigen::VectorXd& x, const Eigen::VectorXd& step,
                       double* trust_region);

 private:
  std::shared_ptr<BinaryFunction> function_ = nullptr;
};

#endif  // SOLVER_LM_H_
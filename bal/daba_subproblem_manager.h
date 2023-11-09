#ifndef BAL_DABA_SUBPROBLEM_MANAGER_H_
#define  BAL_DABA_SUBPROBLEM_MANAGER_H_
#include "bal_solver.h"
#include "first_order_methods/first_oracle.h"
#include <ceres/cost_function.h>

using CameraParameters = std::array<double, 9>;
using PointParameters = std::array<double, 3>;

// A Manager to handle the parameters boardcast between neighbor cluster
// and decide whether restart the nesteorv's accelerated gradient.
class DABASubProblemManager : public ProblemSolver {
  public:
  void Solve(Problem& problem) override;

 private:
  void SetOpmizetionVariable(
      const std::map<int64_t, CameraParameters>& camera_parameters,
      const std::map<int64_t, PointParameters>& point_parameters);


  std::map<int64_t, CameraParameters> camera_parameters_;
  std::map<int64_t, PointParameters> point_parameters_;
  std::map<int64_t, CameraParameters> condition_camera_parameters_;
  std::map<int64_t, PointParameters> condition_point_parameters_;

  std::map<int64_t, CameraParameters> current_camera_parameters_;
  std::map<int64_t, PointParameters> current_point_parameters_;
  std::map<int64_t, CameraParameters> previous_camera_parameters_;
  std::map<int64_t, PointParameters> previous_point_parameters_;
  std::map<int64_t, CameraParameters> auxiliary_camera_parameters_;
  std::map<int64_t, PointParameters> auxiliary_point_parameters_;


};

template<int ParameterNum>
class GradientDescentManager : public FirstOracle {
 private:
  std::vector<ceres::CostFunction*> functions_;
  double* parameters_ = nullptr;
 public:
  //GradientDescentManager(double* parameters) : parameters_(parameters) {}

  void Append(ceres::CostFunction* cost_function) {
    functions_.push_back(cost_function);
  }

  void SetParameters(double* parameters) {
    parameters_ = parameters;
  }

  void Step() {
    return;
    Eigen::Map<Eigen::Matrix<double, ParameterNum, 1>> x0(parameters_);
    Eigen::VectorXd J = this->SubGradient(x0);
    double fval = this->fval(x0);
     double tau = 0.5;
     double r = 1e-4;
     double alpha = 1.0;
     double direction_norm = J.squaredNorm();

     while (fval - this->fval(x0 - alpha * J) < r * alpha * direction_norm) {
       alpha *= tau;
     }
     assert(!std::isnan(J.norm()));
     //std::cout << "Step alpha : " << alpha << std::endl;
     //x0 = x0 - 1e-29 * J;
  }

  Eigen::VectorXd SubGradient(const Eigen::VectorXd &x0) override {
    Eigen::VectorXd J(ParameterNum);
    std::vector<const double*> parameters{x0.data()};
    Eigen::Matrix<double, 3, ParameterNum, Eigen::RowMajor> jacobian;
    std::vector<double*> jacobians = {jacobian.data()};
    for (ceres::CostFunction* f : functions_) {
      Eigen::Vector3d res;
      f->Evaluate(parameters.data(), res.data(), jacobians.data());
      J += res.transpose() * jacobian;
    }
    return J;
  }

  double fval(const Eigen::VectorXd &x0) override {
    double fval = 0.0;
    std::vector<const double*> parameters{x0.data()};
    for (ceres::CostFunction* f : functions_) {
      Eigen::Vector3d res;
      f->Evaluate(parameters.data(), res.data(), nullptr);
      fval += 0.5 * res.squaredNorm();
    }
    return fval;
  }

};

#endif  // BAL_DABA_SUBPROBLEM_MANAGER_H_
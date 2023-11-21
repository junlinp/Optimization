#ifndef BAL_DABA_SUBPROBLEM_MANAGER_H_
#define  BAL_DABA_SUBPROBLEM_MANAGER_H_
#include "bal_solver.h"
#include "first_order_methods/first_oracle.h"

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

class CameraParametersFirstOrderFunction : public FirstOracle {
 std::vector<ceres::CostFunction*> functions_;
 
public:
  void AddCostFunction(ceres::CostFunction* cost_function) {
    functions_.push_back(cost_function);
  }

  double fval(const Eigen::VectorXd &x0) override {
    double f_val = 0;
    
    std::array<const double*, 1> parameters = {
      x0.data()
    };
    for (auto& cost_function : functions_) {
      std::vector<double> res(cost_function->num_residuals());
      cost_function->Evaluate(parameters.data(), res.data(), nullptr);
      for(double e : res) {
        f_val += 0.5 * e * e;
      }
    }
    return f_val;
  }

  Eigen::VectorXd SubGradient(const Eigen::VectorXd &x0) override {}
};

#endif  // BAL_DABA_SUBPROBLEM_MANAGER_H_
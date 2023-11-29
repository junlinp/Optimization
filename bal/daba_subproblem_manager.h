#ifndef BAL_DABA_SUBPROBLEM_MANAGER_H_
#define  BAL_DABA_SUBPROBLEM_MANAGER_H_
#include "bal/bal_solver.h"

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

#endif  // BAL_DABA_SUBPROBLEM_MANAGER_H_
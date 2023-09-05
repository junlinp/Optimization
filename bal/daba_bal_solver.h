#ifndef DABA_BAL_SOLVER_H_
#define DABA_BAL_SOLVER_H_

#include "bal_solver.h"

class DABAProblemSolver : public ProblemSolver {
public:
    void Solve(Problem& problem) override;

private:
    std::map<int64_t, std::array<double, 9>> camera_parameters_;
    std::map<int64_t, std::array<double, 9>> last_camera_parameters_;
    std::map<int64_t, std::array<double, 3>> landmark_position_;
    std::map<int64_t, std::array<double, 3>> last_landmark_position_;

};

#endif // DABA_BAL_SOLVER_H_
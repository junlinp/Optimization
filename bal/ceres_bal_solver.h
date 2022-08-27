#ifndef CERES_BAL_SOLVER_H_
#define CERES_BAL_SOLVER_H_

#include "bal_solver.h"

class CeresProblemSolver : public ProblemSolver {
public:
    void Solve(Problem& problem) override;
};

#endif // CERES_BAL_SOLVER_H_
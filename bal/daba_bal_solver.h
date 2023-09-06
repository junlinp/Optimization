#ifndef DABA_BAL_SOLVER_H_
#define DABA_BAL_SOLVER_H_

#include "bal_solver.h"

class DABAProblemSolver : public ProblemSolver {
public:
    void Solve(Problem& problem) override;


};

#endif // DABA_BAL_SOLVER_H_
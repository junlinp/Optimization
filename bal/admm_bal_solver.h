#ifndef BAL_ADMM_BAL_SOLVER_H_
#define BAL_ADMM_BAL_SOLVER_H_

#include "bal_solver.h"

class ADMMProblemSolver : public ProblemSolver {
public :
    void Solve(Problem&) override;
};
#endif // BAL_ADMM_BAL_SOLVER_H_
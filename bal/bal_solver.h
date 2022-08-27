#ifndef BAL_SOLVER_H_
#define BAL_SOLVER_H_

#include "problem.h"
class ProblemSolver {
public:
    virtual void Solve(Problem& problem) = 0;
};

#endif  // BAL_SOLVER_H_
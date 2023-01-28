#ifndef BAL_DISTRIBUTED_PCG_SOLVER_H_
#define BAL_DISTRIBUTED_PCG_SOLVER_H_
#include "bal_solver.h"
class DistributedPCGSolver : public ProblemSolver {
public:
  void Solve(Problem &problem) override;
};

#endif //  BAL_DISTRIBUTED_PCG_SOLVER_H_
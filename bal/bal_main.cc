#include <chrono>
#include <iostream>
#include <string>

#include "JET.h"
#include "bal/admm_bal_solver.h"
#include "bal/ceres_bal_solver.h"
#include "bal/evaluate.h"
#include "bal/load_problem.h"


int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s /path/to/data_set\n", argv[0]);
    return 0;
  }
  const std::string path = argv[1];

  Problem problem = LoadProblem(path);
  std::cout << "Cameras : " << problem.cameras_.size() << std::endl;
  std::cout << "Points : " << problem.points_.size() << std::endl;
  std::cout << "Observation : " << problem.observations_.size() << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  CeresProblemSolver solver;
  solver.Solve(problem);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << (end - start).count() / 1000.0 / 1000 / 1000 << " seconds."
            << std::endl;
  ADMMProblemSolver admm_solver;
  start = std::chrono::high_resolution_clock::now();
  admm_solver.Solve(problem);
  end = std::chrono::high_resolution_clock::now();
  std::cout << (end - start).count() / 1000.0 / 1000 / 1000 << " seconds."
            << std::endl;

  // ceres::AngleAxisRotatePoint();
  return 0;
}

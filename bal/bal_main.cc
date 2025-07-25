#include <string>
#include "bal/bal_solver.h"
#include "load_problem.h"
#include "iostream"
#include "evaluate.h"
// #include "../JET.h"

#include <chrono>
#include "ceres_bal_solver.h"
#include "daba_bal_solver.h"
#include "daba_subproblem_manager.h"
int main(int argc, char**argv) {
    if(argc < 2) {
	    std::fprintf(stderr, "Usage: %s /path/to/data_set\n", argv[0]);
	    return 0;
    }
    const std::string path = argv[1];


    Problem problem = LoadProblem(path);
    std::cout << "Cameras : " << problem.cameras_.size() << std::endl;
    std::cout << "Points : " << problem.points_.size()  << std::endl;
    std::cout << "Observation : " << problem.observations_.size() << std::endl;
    std::cout << "Problem Origin MSE : " << problem.MSE() << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    std::shared_ptr<ProblemSolver> solver = std::make_shared<DABAProblemSolver>();

    if (argc == 3) {
        if (std::string(argv[2]) == "ceres") {
            solver = std::make_shared<CeresRayProblemSolver>();
        }
        if (std::string(argv[2]) == "manager") {
            solver = std::make_shared<DABASubProblemManager>();
        }
    }
    solver->Solve(problem);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Problem MSE : " << problem.MSE() << std::endl;
    std::cout << (end - start).count() / 1000.0 / 1000 / 1000 << " seconds." << std::endl;
    problem.ToPly("point_cloud.ply");
    return 0;
}

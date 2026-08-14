#include "mps_problem.h"

#include <chrono>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "linear_programing.h"

int main(int argc, char* argv[]) {
#ifdef EIGEN_USE_OPENMP
    Eigen::setNbThreads(omp_get_max_threads());
    std::cout << "Eigen configured to use " << Eigen::nbThreads() << " threads"
              << std::endl;
#else
    std::cout << "Eigen running single-threaded (OpenMP not available)"
              << std::endl;
#endif

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file.mps>\n";
        return 1;
    }
    MPSProblem prob;
    try {
        prob = read_mps(argv[1]);
        std::cout << "Loaded MPS problem: " << prob.name << std::endl;
        std::cout << "Rows: " << prob.row_index.size()
                  << ", Columns: " << prob.col_index.size() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading MPS file: " << e.what() << std::endl;
        return 2;
    }

    StandardFormLp lp = BuildStandardFormLp(prob);
    std::cout << "Standard form rows: " << lp.A.rows()
              << ", columns: " << lp.A.cols()
              << ", nnz: " << lp.A.nonZeros() << std::endl;

    Eigen::VectorXd x = Eigen::VectorXd::Ones(lp.c.size());
    std::cout << "Solving LP with LPSolver3" << std::endl;
    const auto start = std::chrono::steady_clock::now();
    LPSolver3(lp.c, lp.A, lp.b, x);
    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start)
            .count();

    std::cout << "Solved LP" << std::endl;
    std::cout << "Elapsed: " << elapsed_ms << " ms" << std::endl;
    std::cout << "Objective value: " << lp.c.dot(x) + lp.objective_offset << std::endl;
    std::cout << "Primal residual ||Ax - b||: " << (lp.A * x - lp.b).norm()
              << std::endl;
    return 0;
}

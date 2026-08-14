#include "mps_problem.h"

#include <chrono>
#include <iostream>

#include <Eigen/Dense>
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

    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    Eigen::VectorXd c;
    BuildDenseLp(prob, &A, &b, &c);

    Eigen::VectorXd x = Eigen::VectorXd::Ones(c.size());
    std::cout << "Solving LP with LPSolver2" << std::endl;
    const auto start = std::chrono::steady_clock::now();
    LPSolver2(c, A, b, x);
    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start)
            .count();

    std::cout << "Solved LP" << std::endl;
    std::cout << "Elapsed: " << elapsed_ms << " ms" << std::endl;
    std::cout << "Objective value: " << c.dot(x) << std::endl;
    std::cout << "Primal residual ||Ax - b||: " << (A * x - b).norm()
              << std::endl;
    return 0;
}

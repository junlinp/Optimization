#include "mps_problem.h"
#include <iostream>
#include <Eigen/Dense>
#include "linear_programing.h"

int main(int argc, char* argv[]) {
    // Configure Eigen threading
    #ifdef EIGEN_USE_OPENMP
        // Set number of threads for Eigen
        Eigen::setNbThreads(omp_get_max_threads());
        std::cout << "Eigen configured to use " << Eigen::nbThreads() << " threads" << std::endl;
        std::cout << "OpenMP max threads: " << omp_get_max_threads() << std::endl;
    #else
        std::cout << "Eigen running single-threaded (OpenMP not available)" << std::endl;
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

    // Build the constraint matrix (A), rhs (b), and objective (c)
    size_t m = prob.row_index.size();
    size_t n = prob.col_index.size();

    Eigen::MatrixXd A(m, n);
    for (auto it = prob.coefficients.begin(); it != prob.coefficients.end(); ++it) {
        for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
            A(it->first, it2->first) = it2->second;
        }
    }


    Eigen::VectorXd b(m);
    for (auto it = prob.rhs.begin(); it != prob.rhs.end(); ++it) {
        b[it->first] = it->second;
    }
    Eigen::VectorXd c(n);
    for (auto it = prob.objective_row_coefficients.begin(); it != prob.objective_row_coefficients.end(); ++it) {
        c[it->first] = it->second;
    }

    Eigen::VectorXd x(n);
    x.setRandom();
    // data from
    // https://plato.asu.edu/ftp/lptestset/
    // 
    std::cout << "Trivial solution (all variables at lower bound):\n";
    std::cout << "Objective value: " << c.transpose() * x << std::endl;
    std::cout << "Solving LP" << std::endl;
    // LPSolver don't converge for this problem
    // LPSolver(c, A, b, x);
    LPSolver2(c, A, b, x);
    std::cout << "Solved LP" << std::endl;
    std::cout << "Objective value: " << c.transpose() * x << std::endl;
    // std::cout << "Solution: " << x.transpose() << std::endl;
    return 0;
}

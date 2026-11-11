#include "mps_problem.h"

#include <chrono>
#include <iostream>
#include <iomanip>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "jordan_conic.h"

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

    std::cout << "Solving LP with sparse Jordan backend" << std::endl;
    const auto start = std::chrono::steady_clock::now();
    jordan::Options options;
    options.max_iterations = 400;
    options.tolerance = 1e-6;
    const jordan::Result result = jordan::SolveSparse(
        lp.A, lp.b, lp.c, {{jordan::ConeType::Nonnegative, static_cast<int>(lp.c.size())}}, options);
    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start)
            .count();

    const char* status = "invalid input";
    switch (result.status) {
      case jordan::Status::Optimal: status = "optimal"; break;
      case jordan::Status::MaxIterations: status = "iteration limit"; break;
      case jordan::Status::NumericalFailure: status = "numerical failure"; break;
      case jordan::Status::InvalidInput: break;
    }
    std::cout << "Status: " << status << " (iterations: " << result.iterations << ")" << std::endl;
    std::cout << "Elapsed: " << elapsed_ms << " ms" << std::endl;
    if (result.x.size() == lp.c.size()) {
      std::cout << std::setprecision(12) << "Objective value: " << lp.c.dot(result.x) + lp.objective_offset << std::endl;
      std::cout << "Primal residual ||Ax - b||: " << result.primal_residual << std::endl;
      std::cout << "Dual residual ||c - A^T y - s||: " << result.dual_residual << std::endl;
      std::cout << "Complementarity gap: " << result.gap << std::endl;
    }
    return result.status == jordan::Status::Optimal ? 0 : 3;
}

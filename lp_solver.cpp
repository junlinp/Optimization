#include <iostream>
#include "IO/mps_format_io.h"
#include "linear_programing.h"
#include "Eigen/SparseQR"
#include "Eigen/IterativeLinearSolvers"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("Usage: %s mpsfile\n", argv[0]);
        return 0;
    }
    Problem problem;
    LoadMPSProblem(argv[1], problem);
    Eigen::VectorXd c;
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd b;

    ConstructProblem(problem, c, A, b);
    std::cout << "A[" << A.rows() << "," << A.cols() << "] nnz  : " << A.nonZeros() << std::endl;
    Eigen::VectorXd x;
    using namespace Eigen;
    //ConjugateGradient<SparseMatrix<double>, COLAMDOrdering<int>> solver;
    LeastSquaresConjugateGradient<SparseMatrix<double>> solver;
    solver.compute(A);
    std::cout << (solver.info() == Success) << std::endl;
    x = solver.solve(b);
    
    std::cout << "#iterations:     " << solver.iterations() << std::endl;
    std::cout << "estimated error: " << solver.error()      << std::endl;
    //PCVI2(c, A, b, x);
    //SparsePrimDualLogarithmSolver(c, A, b, x);
    ADMM(c, A, b, x);


}
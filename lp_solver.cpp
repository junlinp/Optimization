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
    std::cout << "ConstructProblem Finish" << std::endl;
    Eigen::VectorXd x;
    
    using namespace Eigen;
    //ConjugateGradient<SparseMatrix<double>, COLAMDOrdering<int>> solver;
    ConjugateGradient<SparseMatrix<double>> solver;
    solver.compute(SparseMatrix<double>(A.transpose()) * A);
    std::cout << (solver.info() == Success) << std::endl;
    Eigen::VectorXd ATb = A.transpose() * b;
    std::cout << "AT b : " << std::endl;
    x = solver.solve(ATb);
    std::cout << "Constraint Norm : " << (A * x - b).norm() << std::endl;
    
    //PCVI(c, A, b, x);


}
#include "linear_programing.h"
#include <chrono>
#include <memory>

namespace optimized {

// Optimized SDP solver with performance improvements
class OptimizedSDPSolver {
private:
    Eigen::SparseMatrix<double> A_;
    Eigen::VectorXd b_;
    Eigen::VectorXd C_;
    std::vector<Eigen::SparseMatrix<double>> mat_A_;
    
    // Cached solver and factorization
    std::unique_ptr<Eigen::SparseLU<Eigen::SparseMatrix<double>>> solver_;
    Eigen::SparseMatrix<double> last_coefficient_matrix_;
    bool solver_initialized_ = false;
    
    // Cached computations
    Eigen::VectorXd last_w_;
    bool w_cached_ = false;
    
    // Parameters
    double zeta_ = 5.0;
    double mu0_;
    double epsilon_ = 1e-7;
    double theta_;
    double delta_ = 1.0;

public:
    OptimizedSDPSolver(const Eigen::VectorXd& C, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b)
        : A_(A), b_(b), C_(C) {
        
        // Pre-compute mat_A_ once
        Eigen::SparseMatrix<double, Eigen::RowMajor> R_A(A);
        mat_A_.reserve(A.rows());
        for(int i = 0; i < A.rows(); i++) {
            mat_A_.push_back(ToMatrix(R_A.row(i)));
        }
        
        // Initialize parameters
        mu0_ = zeta_ * zeta_;
        theta_ = 0.25 / std::sqrt(A.cols());
        
        // Initialize solver
        solver_ = std::make_unique<Eigen::SparseLU<Eigen::SparseMatrix<double>>>();
    }
    
    void solve(Eigen::VectorXd& X) {
        using namespace Eigen;
        
        // Initialize variables
        X = SemiDefineSpace::IdentityWithPurterbed(X.rows(), zeta_);
        VectorXd S = X;
        VectorXd y(A_.rows());
        y.setZero();
        
        VectorXd X0 = X, S0 = S;
        VectorXd y0 = y;
        size_t epoch = 0;
        
        VectorXd rb = b_ - A_ * X0;
        VectorXd rc = C_ - A_.transpose() * y0 - S0;
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
        while (Max(SemiDefineSpace::Trace(X, S), (A_*X - b_).norm(), 
                   SemiDefineSpace::Norm(C_ - A_.transpose() * y - S)) > epsilon_) {
            
            auto epoch_start = std::chrono::high_resolution_clock::now();
            std::printf("Epoch %zu\t<C, X>=[%.6f]\tPrim Constraint[%.6f]\tDual Constraint[%.6f]\n", 
                       ++epoch, C_.dot(X), (A_*X-b_).norm(), 
                       SemiDefineSpace::Norm(C_-A_.transpose() * y - S));
            
            // Optimized feasible step
            auto [delta_x, delta_y, delta_s] = optimizedFeasibleStep(X, S, y, rb, rc);
            
            X += delta_x;
            y += delta_y;
            S += delta_s;
            delta_ = (1 - theta_) * delta_;
            
            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
            
            if (epoch % 10 == 0) {
                std::printf("Epoch %zu took: %ld ms\n", epoch, epoch_duration.count());
            }
            
            // Check feasibility
            if (!SemiDefineSpace::Varify(X)) {
                std::printf("Warning: X is infeasible\n");
            }
            if (!SemiDefineSpace::Varify(S)) {
                std::printf("Warning: S is infeasible\n");
            }
        }
        
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
        std::printf("Total solving time: %ld ms\n", total_duration.count());
    }
    
private:
    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> 
    optimizedFeasibleStep(const Eigen::VectorXd& X, const Eigen::VectorXd& S, 
                         const Eigen::VectorXd& y, const Eigen::VectorXd& rb, 
                         const Eigen::VectorXd& rc) {
        using namespace Eigen;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Compute w with caching
        VectorXd w = computeW(X, S);
        VectorXd sqrt_w = SemiDefineSpace::Sqrt(w);
        VectorXd inv_sqrt_w = SemiDefineSpace::Inverse(sqrt_w);
        
        double mu = delta_ * mu0_;
        double inverse_sqrt_mu = 1.0 / (std::sqrt(mu) + std::numeric_limits<double>::epsilon());
        
        VectorXd v = inverse_sqrt_mu * SemiDefineSpace::P(sqrt_w, S, sqrt_w);
        VectorXd prim = theta_ * delta_ * rb;
        VectorXd dual = inverse_sqrt_mu * theta_ * delta_ * SemiDefineSpace::P(sqrt_w, sqrt_w, rc);
        VectorXd comp = ((1 - theta_) * SemiDefineSpace::Inverse(v) - v);
        
        auto build_end = std::chrono::high_resolution_clock::now();
        std::cout << "Build time: " << (build_end - start).count() / 1000000 << " ms" << std::endl;
        
        // Optimized linear solving
        VectorXd dy = solveLinearSystem(w, prim, dual, comp, mu, sqrt_w);
        
        VectorXd dx = inverse_sqrt_mu * SemiDefineSpace::P(sqrt_w, sqrt_w, A_.transpose() * dy) - dual + comp;
        VectorXd ds = dual - inverse_sqrt_mu * SemiDefineSpace::P(sqrt_w, sqrt_w, A_.transpose() * dy);
        
        auto solve_end = std::chrono::high_resolution_clock::now();
        std::cout << "Solve time: " << (solve_end - build_end).count() / 1000000 << " ms" << std::endl;
        
        VectorXd delta_x = std::sqrt(mu) * SemiDefineSpace::P(sqrt_w, dx, sqrt_w);
        VectorXd delta_s = std::sqrt(mu) * SemiDefineSpace::P(inv_sqrt_w, ds, inv_sqrt_w);
        
        return {delta_x, dy, delta_s};
    }
    
    Eigen::VectorXd computeW(const Eigen::VectorXd& X, const Eigen::VectorXd& S) {
        // Check if we can reuse cached w
        if (w_cached_ && (X - last_w_).norm() < 1e-6) {
            return last_w_;
        }
        
        Eigen::VectorXd X_sqrt = SemiDefineSpace::Sqrt(X);
        Eigen::VectorXd temp = SemiDefineSpace::Sqrt(SemiDefineSpace::Inverse(SemiDefineSpace::P(X_sqrt, X_sqrt, S)));
        Eigen::VectorXd w = SemiDefineSpace::P(X_sqrt, X_sqrt, temp);
        
        // Cache for next iteration
        last_w_ = w;
        w_cached_ = true;
        
        return w;
    }
    
    Eigen::VectorXd solveLinearSystem(const Eigen::VectorXd& w, const Eigen::VectorXd& prim, 
                                     const Eigen::VectorXd& dual, const Eigen::VectorXd& comp,
                                     double mu, const Eigen::VectorXd& sqrt_w) {
        using namespace Eigen;
        
        // Check if we need to recompute the coefficient matrix
        bool need_new_factorization = !solver_initialized_;
        
        if (!need_new_factorization) {
            // Simple heuristic: recompute if w changed significantly
            double w_change = (w - last_w_).norm();
            need_new_factorization = (w_change > 0.1);
        }
        
        if (need_new_factorization) {
            auto factor_start = std::chrono::high_resolution_clock::now();
            
            SparseMatrix<double> coefficient = ComputeADAT(mat_A_, w);
            solver_->compute(coefficient.selfadjointView<Upper>());
            
            auto factor_end = std::chrono::high_resolution_clock::now();
            std::cout << "Factorization time: " << (factor_end - factor_start).count() / 1000000 << " ms" << std::endl;
            
            solver_initialized_ = true;
            last_w_ = w;
        }
        
        // Solve the linear system
        VectorXd rhs = prim + std::sqrt(mu) * (A_ * SemiDefineSpace::P(sqrt_w, sqrt_w, (dual - comp)));
        return solver_->solve(rhs);
    }
    
    // Helper function to compute max of three values
    double Max(double a, double b, double c) {
        return std::max(std::max(a, b), c);
    }
};

// Optimized version of ComputeADAT with better performance
Eigen::SparseMatrix<double> ComputeADATOptimized(const std::vector<Eigen::SparseMatrix<double>>& mat_A, 
                                                const Eigen::VectorXd& w) {
    Eigen::MatrixXd Mat_w = SemiDefineSpace::Mat(w);
    std::vector<Eigen::MatrixXd> Mat_wAw;
    Mat_wAw.reserve(mat_A.size());
    
    // Pre-compute all Mat_wAw products
    #pragma omp parallel for
    for(size_t i = 0; i < mat_A.size(); i++) {
        Mat_wAw.push_back(Mat_w * mat_A[i] * Mat_w);
    }
    
    Eigen::SparseMatrix<double> res(mat_A.size(), mat_A.size());
    using T = Eigen::Triplet<double>;
    std::vector<T> triplet;
    triplet.reserve(mat_A.size() * (mat_A.size() + 1) / 2); // Upper triangular
    
    // Compute only upper triangular part
    for(size_t col = 0; col < mat_A.size(); col++) {
        for(size_t row = 0; row <= col; row++) {
            double value = mat_A[row].cwiseProduct(Mat_wAw[col]).sum();
            triplet.push_back(T{static_cast<int>(row), static_cast<int>(col), value});
        }
    }
    
    res.setFromTriplets(triplet.begin(), triplet.end());
    return res;
}

} // namespace optimized

// Optimized SDP solver interface
void OptimizedSDPSolver(const Eigen::VectorXd& C, const Eigen::SparseMatrix<double>& A, 
                       const Eigen::VectorXd& b, Eigen::MatrixXd& X) {
    Eigen::VectorXd x(X.rows() * X.cols());
    std::cout << "Optimized SDP Solver" << std::endl;
    
    optimized::OptimizedSDPSolver solver(C, A, b);
    solver.solve(x);
    
    std::cout << "Optimized SDP Solver Finished" << std::endl;
    X = SemiDefineSpace::Mat(x);
} 
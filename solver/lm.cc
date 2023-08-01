#include "lm.h"
#include "iostream"

double QuadraticSolve(const Eigen::VectorXd& s, const Eigen::VectorXd& d, double trust_region) {

    double a = d.dot(d);
    double b = 2.0 * s.dot(d);
    double c = s.dot(s) - trust_region * trust_region;

    return (-b + std::sqrt(b * b - 4.0 * a * c)) / a / 2.0;
}

void TRSSolve(const Eigen::MatrixXd& J, const Eigen::VectorXd& e, double trust_region, Eigen::VectorXd* step) {
    // TODO(junlinp): B can be optimization
    Eigen::VectorXd s = Eigen::VectorXd::Zero(e.cols());
    std::cout << "s" << std::endl;
    Eigen::VectorXd g = J.transpose() * e;
    std::cout << "g" << std::endl;
    Eigen::MatrixXd B = J.transpose() * J;
    std::cout << "B" << std::endl;
    Eigen::VectorXd d = -g;
    std::cout << "d" << std::endl;
    std::cout << "g row and col : " << g.rows() << ", " << g.cols() << std::endl;
    double sum = 0.0;
    for(int i = 0; i < g.rows(); i++) {
        sum += g(i) * g(i);
    }
    std::cout << "sum : " << sum << std::endl;
    // Truncated CG
    while(g.norm() < 1e-6) {
        std::cout << "last_square_norm_" << std::endl;
        double last_square_norm_g = g.dot(g);
        double dT_B_d = d.dot(B * d);
        if (dT_B_d< 0) {
            // alpha should be larger then 0
            double step_size = QuadraticSolve(s, d, trust_region);
            *step = s + step_size * d;
            return;
        }
        double alpha = last_square_norm_g / dT_B_d;

        Eigen::VectorXd new_s = s + alpha * d;
        if (new_s.dot(new_s) > trust_region * trust_region) {
            double step_size = QuadraticSolve(s, d, trust_region);
            *step = s + step_size * d;
            return;
        } else {
            s = new_s;
        }
        g = g + alpha * B * d;
        double beta = g.dot(g) / last_square_norm_g;
        d = beta * d - g;
    }
    *step  = s;
}

bool LMSolver::Solve(Eigen::VectorXd *x) {
    Eigen::VectorXd x0 = *x;
    size_t max_iteration = 1024;
    size_t iterator = 0;
    double trust_region = 1.0;
    while(iterator++ < max_iteration) {
        Eigen::MatrixXd J = function_->Jacobians(x0);
        Eigen::VectorXd e = function_->Evaluate(x0);
        std::cout << "Evaluate " << std::endl;
        Eigen::VectorXd step;
        TRSSolve(J, e, trust_region, &step);
        // need function to evaluate the next step
        if (AcceptStepOrNot(x0, step, &trust_region)) {
            x0 = x0 + step;
        }
    }
    *x = x0;
    return true;
}

bool LMSolver::AcceptStepOrNot(const Eigen::VectorXd &x, const Eigen::VectorXd &step, double *trust_region) {
    const double eta_v = 0.9;
    const double eta_s = 0.1;

    Eigen::VectorXd fx = function_->Evaluate(x);
    Eigen::MatrixXd J = function_->Evaluate(x);
    Eigen::VectorXd fx_next = function_->Evaluate(x + step);

    double rho = (fx.dot(fx) - fx_next.dot(fx_next)) /
                 (fx.dot(fx) - (fx + J * step).squaredNorm());

    if (rho > eta_v) {
        *trust_region *= 2;
        return true;
    } else if (rho >= eta_s) {
        return true;
    } else {
        *trust_region *= 0.5;
        return false;
    }
}


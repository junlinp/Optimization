#include "lm.h"
#include "iostream"

double QuadraticSolve(const Eigen::VectorXd& s, const Eigen::VectorXd& d, double trust_region) {

    double a = d.dot(d);
    double b = 2.0 * s.dot(d);
    double c = s.dot(s) - trust_region * trust_region;

    return (-b + std::sqrt(b * b - 4.0 * a * c)) / a / 2.0;
}

void TRSSolve(const Eigen::MatrixXd& J, const Eigen::VectorXd& e, double trust_region, Eigen::VectorXd* step) {
    // TODO(junlinp): precondition
    // TODO(junlinp): B can be optimization
    Eigen::VectorXd s = Eigen::VectorXd::Zero(e.rows());
    Eigen::VectorXd g = J.transpose() * e;
    Eigen::MatrixXd B = J.transpose() * J;
    Eigen::VectorXd d = -g;
 
    // Truncated CG
    while(g.norm() > 1e-6) {
        double last_square_norm_g = g.dot(g);
        double dT_B_d = d.dot(B * d);
        if (dT_B_d < 0) {
            // alpha should be larger then 0
            double step_size = QuadraticSolve(s, d, trust_region);
            *step = s + step_size * d;
            return;
        }
        double alpha = last_square_norm_g / dT_B_d;

        Eigen::VectorXd new_s = s + alpha * d;
        if (new_s.squaredNorm() > trust_region * trust_region) {
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
        std::cout << "Evaluate :" << 0.5 * e.dot(e) << std::endl;
        Eigen::VectorXd step = Eigen::VectorXd::Zero(function_->VariableDimension());
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
    std::cout << "AcceptStepOrNot" << std::endl;
    Eigen::VectorXd fx = function_->Evaluate(x);
    std::cout << "Evaluate fx" << std::endl;
    Eigen::MatrixXd J = function_->Jacobians(x);
    std::cout << "Evaluate J" << std::endl;
    Eigen::VectorXd fx_next = function_->Evaluate(x + step);
    std::cout << "Evaluate fx_next" << std::endl;
    std::cout << "fx_next row : " << fx_next.rows() << std::endl;

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
    return false;
}


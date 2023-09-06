#include "lm.h"
#include "iostream"

double QuadraticSolve(const Eigen::VectorXd& s, const Eigen::VectorXd& d, double trust_region) {

    double a = d.dot(d);
    double b = 2.0 * s.dot(d);
    double c = s.dot(s) - trust_region * trust_region;

    return (-b + std::sqrt(b * b - 4.0 * a * c)) / a / 2.0;
}
bool IncompleteCholesky(const Eigen::MatrixXd& A, Eigen::MatrixXd* L) {
    // LLT = A
    assert(A.rows() == A.cols());
    size_t n = A.rows();
    for (size_t i = 0; i < n; i++) {
        (*L)(i, i) = A(i, i);
        for (int j = 0; j < i; j++) {
            (*L)(i, i) -= std::pow((*L)(i, j), 2.0);
        }

        (*L)(i, i) = std::sqrt((*L)(i, i));

        double invert = 1.0 / (*L)(i, i);
        for (int j = i + 1; j < n; j++) {
            double t = A(j, i);
            for (int k = 0; k < i; k++) {
                t -= (*L)(i, k) * (*L)(j, k);
            }
            (*L)(j, i) = invert * t;
        }
    }
    return true;
}

bool JacobiPreconditioner(const Eigen::MatrixXd& A, Eigen::MatrixXd* L) {
    assert(A.rows() == A.cols());
    size_t n = A.rows();
    for(size_t i = 0; i < n; i++) {
        (*L)(i, i) = std::sqrt(A(i, i));
    }
    return true;
}

void TRSSolve(const Eigen::MatrixXd& J, const Eigen::VectorXd& e, double trust_region, Eigen::VectorXd* step) {
    // TODO(junlinp): precondition
    // TODO(junlinp): B can be optimization
    Eigen::VectorXd s = Eigen::VectorXd::Zero(e.rows());
    Eigen::MatrixXd B = J.transpose() * J;
    
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(B.rows(), B.cols());
    //IncompleteCholesky(B, &L);
    JacobiPreconditioner(B, &L);

    B = L.inverse() * B * L.transpose().inverse();
    Eigen::VectorXd g = J.transpose() * e;
    g = L.inverse() * g;
    Eigen::VectorXd d = -g;
    size_t iterator = 0;
    // Truncated CG
    while(g.norm() > 1e-6) {
        double last_square_norm_g = g.dot(g);
        Eigen::VectorXd B_d = B*d;
        double dT_B_d = d.dot(B_d);
        if (dT_B_d < 0) {
            // alpha should be larger then 0
            double step_size = QuadraticSolve(s, d, trust_region);
            *step = s + step_size * d;
            *step = L.transpose().inverse() * (*step);
            std::cout << "Iterator CG  : " << iterator << std::endl;
            return;
        }
        double alpha = last_square_norm_g / dT_B_d;

        Eigen::VectorXd new_s = s + alpha * d;
        if (new_s.squaredNorm() > trust_region * trust_region) {
            double step_size = QuadraticSolve(s, d, trust_region);
            *step = s + step_size * d;
            *step = L.transpose().inverse() * (*step);
            std::cout << "Iterator CG  : " << iterator << std::endl;
            return;
        } else {
            s = new_s;
        }
        g = g + alpha * B_d;
        double beta = g.dot(g) / last_square_norm_g;
        d = beta * d - g;
        // std::cout << "s: " << s.norm() << std::endl;
        iterator++;
    }
    *step  = s;
    *step = L.transpose().inverse() * (*step);
    std::cout << "Iterator CG  : " << iterator << std::endl;
}



bool LMSolver::Solve(Eigen::VectorXd *x) {
    Eigen::VectorXd x0 = *x;
    size_t max_iteration = 1024;
    size_t iterator = 0;

    Eigen::MatrixXd J = function_->Jacobians(x0);
    Eigen::VectorXd e = function_->Evaluate(x0);
    double trust_region = ((J.transpose() * J).inverse() * J * e).norm();
    std::cout << "Trust Region : " << trust_region << std::endl;
    while(iterator++ < max_iteration) {
        Eigen::MatrixXd J = function_->Jacobians(x0);
        Eigen::VectorXd e = function_->Evaluate(x0);
        std::cout << "Evaluate :" << 0.5 * e.dot(e) << std::endl;
        if (0.5 * e.dot(e) < 1e-13) {
            break;
        }
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


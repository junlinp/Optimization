#include "RGD/conjugate_gradient.h"
#include <iostream>

bool ConjugateGradient(const Eigen::MatrixXd&A, const Eigen::VectorXd& b, double tolerance, Eigen::VectorXd* solution) {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());
    Eigen::VectorXd residuals = b - A * x;
    Eigen::VectorXd d = residuals;

    for (int i = 0; i < b.size(); i++) {
        double previous_norm = residuals.squaredNorm();
        double alpha = previous_norm / d.dot(A *d);

        x = x + alpha * d;
        residuals = residuals - alpha * A * d;

        double beta = residuals.squaredNorm() / previous_norm;
        d = residuals + beta * d;
    }
    *solution = x;
    return true;
}
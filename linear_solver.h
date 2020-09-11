#ifndef OPTIMIZATION_LINEAR_SOLVER_H_
#define OPTIMIZATION_LINEAR_SOLVER_H_
#include "Eigen/Dense"


template<class Matrix, class Vector>
void ConjugateGradient(const Matrix&A,const Vector& b, Vector& x) {
    Vector r = b - A * x;
    Vector p = r;

    int max_iterator = 8196;
    int iterator = 0;
    double last_r_dot_r = r.dot(r);
    while(iterator++ < max_iterator) {
        if (r.norm() < 1e-5 * r.rows()) {return;}
        Vector temp = A * p;
        double alpha = last_r_dot_r / p.dot(temp);
        x = x + alpha * p;
        r = r - alpha * temp;
        double new_r_dot_r = r.dot(r);
        double beta = new_r_dot_r / last_r_dot_r;
        p = r + beta * p;
        last_r_dot_r = new_r_dot_r;
    }
}

#endif  // OPTIMIZATION_LINEAR_SOLVER_H_
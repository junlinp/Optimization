#include "lm.h"
#include "iostream"
#include "Eigen/Sparse"
#include <limits>

double QuadraticSolve(const Eigen::VectorXd& s, const Eigen::VectorXd& d, double trust_region) {

    double a = d.dot(d);
    double b = 2.0 * s.dot(d);
    double c = s.dot(s) - trust_region * trust_region;
    std::cout << b * b - 4.0 * a * c << std::endl;

    if (b * b - 4.0 * a * c < 0.0f) {
        std::cout << "Quadratic not solution" << std::endl;
        std::cout << "a, b, c : " << a << "," << b << "," << c << std::endl;
        return 0.0;
    }
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
    size_t max_iteration = 1e6;
    // Truncated CG
    while(g.norm() > 1e-10 && iterator < max_iteration) {
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
    std::cout << "norm g : " << g.norm() << std::endl;
    *step  = s;
    *step = L.transpose().inverse() * (*step);
    std::cout << "Iterator CG  : " << iterator << std::endl;
}



bool LMSolver::Solve(Eigen::VectorXd *x) {
    Eigen::VectorXd x0 = *x;
    size_t max_iteration = 1024;
    double loss_diff_tolerance = 1e-7;
    size_t iterator = 0;
    size_t last_error = std::numeric_limits<double>::max() - 1;

    Eigen::MatrixXd J = function_->Jacobians(x0);
    Eigen::VectorXd e = function_->Evaluate(x0);
    double trust_region = ((J.transpose() * J).inverse() * J * e).norm();
    std::cout << "Trust Region : " << trust_region << std::endl;
    while(iterator++ < max_iteration) {
        Eigen::MatrixXd J = function_->Jacobians(x0);
        Eigen::VectorXd e = function_->Evaluate(x0);

        double loss = 0.5 * e.dot(e);
        std::cout << "Evaluate :" << loss << std::endl;
        if (loss < 1e-13 || std::abs(last_error - loss) < loss_diff_tolerance) {
            if (loss < 1e-13) {
                std::cout << "terminal due to loss [" << loss << "] < " << 1e-13 << std::endl;
            } else {
                std::cout << "terminal due to difference of loss [" << std::abs(last_error - loss) << "] < " << loss_diff_tolerance << std::endl;
            }
            break;
        } else {
            last_error = loss;
        }
        Eigen::VectorXd step = Eigen::VectorXd::Zero(function_->VariableDimension());
        TRSSolve(J, e, trust_region, &step);
        // need function to evaluate the next step
        if (AcceptStepOrNot(x0, step, &trust_region)) {
          x0 = x0 + step;
            std::cout << "Accept step with norm : " << step.norm() << std::endl;
        } else {
            std::cout << "Reject" << std::endl;
        }
    }
    *x = x0;
    return true;
}

bool LMSolver::AcceptStepOrNot(const Eigen::VectorXd &x, const Eigen::VectorXd &step, double *trust_region) {
    const double eta_v = 0.9;
    const double eta_s = 0.1;
    Eigen::VectorXd fx = function_->Evaluate(x);
    Eigen::MatrixXd J = function_->Jacobians(x);
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
    return false;
}

using SPM = Eigen::SparseMatrix<double>;
template<class T>
void TRSSolve(const T& Jc, const T& Jp, const Eigen::VectorXd& fval,
              double trust_region, Eigen::VectorXd* delta_c,
              Eigen::VectorXd* delta_p) {
  //
  // Construct V
  // [Jc^T Jc   Jc^T Jp ]  [delta_c]   =  -[Jc^T f]
  // [ Jp^T Jc   Jp^T Jp]  [delta_p]       [Jp^T f]
  size_t camera_size = Jc.rows();
  size_t point_size = Jp.rows();

  Eigen::VectorXd s = Eigen::VectorXd::Zero(camera_size + point_size);

  Eigen::VectorXd g(camera_size + point_size);
  g.block(0, 0, camera_size, 1).noalias() = Jc.transpose() * fval;
  g.block(camera_size, 0, point_size, 1).noalias() = Jp.transpose() * fval;
  Eigen::VectorXd d = -g;

  size_t iterator = 0;
  size_t max_iteration = 128;
  // Truncated CG
  while (g.norm() > 1e-10 && iterator < max_iteration) {
    double last_square_norm_g = g.dot(g);
    Eigen::VectorXd dc = d.block(0, 0, camera_size, 1);
    Eigen::VectorXd dp = d.block(camera_size, 0, point_size, 1);
    Eigen::VectorXd Jdc = Jc * dc;
    Eigen::VectorXd Jdp = Jp * dp;
    double dT_B_d = Jdc.dot(Jdc) + 2.0 * Jdc.dot(Jdp) + Jdp.dot(Jdp);

    if (dT_B_d < 0) {
      // alpha should be larger then 0
      double step_size = QuadraticSolve(s, d, trust_region);
      Eigen::VectorXd next_step = s + step_size * d;
      *delta_c = next_step.block(0, 0, camera_size, 1);
      *delta_p = next_step.block(camera_size, 0, point_size, 1);
      //*step = L.transpose().inverse() * (*step);
      std::cout << "Iterator CG  : " << iterator << std::endl;
      std::cout << "CG terminal since non-SPD" << std::endl;
      return;
    }
    double alpha = last_square_norm_g / dT_B_d;

    Eigen::VectorXd new_s = s + alpha * d;
    if (new_s.squaredNorm() > trust_region * trust_region) {
      double step_size = QuadraticSolve(s, d, trust_region);
      Eigen::VectorXd next_step = s + step_size * d;
      std::cout << step_size << std::endl;
      std::cout << "d norm " << d.norm() << std::endl;
      std::cout << "next_step : " << next_step.norm() << std::endl;
      *delta_c = next_step.block(0, 0, camera_size, 1);
      *delta_p = next_step.block(camera_size, 0, point_size, 1);
      //*step = L.transpose().inverse() * (*step);
      std::cout << "Iterator CG  : " << iterator << std::endl;
      std::cout << "CG terminal since reach trust region limits" << std::endl;
      return;
    } else {
      s = new_s;
    }
    // g = g + alpha * B_d;
    g.block(0, 0, camera_size, 1) +=
        alpha * Jc.transpose() * (Jc * dc + Jp * dp);
    g.block(camera_size, 0, point_size, 1) +=
        alpha * (Jp.transpose() * (Jc * dc + Jp * dp));

    double beta = g.dot(g) / last_square_norm_g;
    d = beta * d - g;
    // std::cout << "s: " << s.norm() << std::endl;
    iterator++;

  }
  *delta_c = s.block(0, 0, camera_size, 1);
  *delta_p = s.block(camera_size, 0, point_size, 1);
  std::cout << "CG terminal since reach max iteration " << max_iteration << " limits" << std::endl;
}


bool BundleAdjustmentLMSolver::Solve(Eigen::VectorXd* x) {

    Eigen::VectorXd x0 = *x;
    size_t max_iteration = 1024;
    size_t iterator = 0;

    double trust_region = 1e6;

    while(iterator++ < max_iteration) {
        Eigen::MatrixXd first_jacobians, second_jacobians;
        function_->BinaryJacobians(&first_jacobians, &second_jacobians);
        Eigen::VectorXd e = function_->Evaluate(x0);

        Eigen::VectorXd first_delta, second_delta;
        TRSSolve(first_jacobians, second_jacobians, e, trust_region, &first_delta, &second_delta);
        Eigen::VectorXd delta(first_delta.rows() + second_delta.rows());
        delta << first_delta, second_delta;

        if (AcceptStepOrNot(x0, delta, &trust_region)) {
            x0 += delta;
        }
    }
    *x = x0;
    return true;
}

bool BundleAdjustmentLMSolver::AcceptStepOrNot(const Eigen::VectorXd &x, const Eigen::VectorXd &step, double *trust_region) {
    const double eta_v = 0.9;
    const double eta_s = 0.1;
    Eigen::VectorXd fx = function_->Evaluate(x);
    Eigen::MatrixXd J = function_->Jacobians(x);
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
    return false;
}

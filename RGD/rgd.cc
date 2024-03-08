#include "rgd.h"
#include "Eigen/Householder"
#include "iostream"
#include "rgd_cost_function_interface.h"
#include "so3_cost_function_interface.h"
#include <memory>

auto New_X(const std::vector<SO3Manifold::Vector>& x, const std::vector<SO3Manifold::TangentVector>& steps) {
    std::vector<SO3Manifold::Vector> res;
    for (size_t i = 0; i < x.size(); i++) {
      res.push_back(SO3Manifold::Retraction(x[i], steps[i]));
    }
    return res;
}

auto NewStep(const std::vector<SO3Manifold::TangentVector> &directions,
             double step_size) {
  std::vector<SO3Manifold::TangentVector> res;
  for (SO3Manifold::TangentVector tangent_vector : directions) {
    res.push_back(step_size * tangent_vector);
  }
  return res;
}

double BackTracing(const SO3CostFunctionInterface& cost_function, const std::vector<SO3Manifold::Vector>& x ,
const std::vector<SO3Manifold::TangentVector>& directions
) {
  double tau = 0.8;
  double r = 1e-4;
  double alpha = 1.0;
  double direction_norm = 0.0;
  for (const SO3Manifold::TangentVector& v : directions) {
    direction_norm += v.squaredNorm();
  }


  while (cost_function.Evaluate(x) - cost_function.Evaluate(New_X(x, NewStep(directions, alpha))) < r * alpha * direction_norm) {
    alpha *= tau;
  }
  return alpha;
}

double RGDBackTracking(const std::shared_ptr<RGDFirstOrderInterface>& cost_function, const Eigen::VectorXd& x,const Eigen::VectorXd& direction) {
  double tau = 0.8;
  double r = 1e-8;
  double alpha = 1.0;
  double direction_norm = direction.squaredNorm();
  int max_iteration = 1024;
  int iteration = 0;
  while (
      cost_function->Evaluate(x) -
          cost_function->Evaluate(cost_function->Move(x, alpha * direction)) <
      r * alpha * direction_norm && iteration++ < max_iteration) {
    alpha *= tau;
  }
  if (iteration >= max_iteration) {
    std::cout << "Warning : Iteration reach Max Iteration" << std::endl;
    return 0.0;
  }
  std::cout << "BackTracing Initial Cost : " << cost_function->Evaluate(x) << std::endl;
  std::cout << "BackTracing Move Cost : " << cost_function->Evaluate(cost_function->Move(x, alpha * direction)) << std::endl;
  std::cout << "r * alpha * direction_norm : " << r * alpha * direction_norm << std::endl;
  std::cout << "iteration : " << iteration << std::endl;

  return alpha;
}

bool rgd(const SO3CostFunctionInterface &cost_function,
         std::vector<SO3Manifold::Vector> *x_init) {
  size_t max_iteration = 128;
  size_t iteration = 0;
  std::cout << "Initial error : " << cost_function.Evaluate(*x_init) << std::endl;
  while (iteration++ < max_iteration) {
    auto jacobians = cost_function.Jacobian(*x_init);
    std::vector<SO3Manifold::TangentVector> directions;
    for (size_t i = 0; i < x_init->size(); i++) {
      SO3Manifold::TangentVector TxU = SO3Manifold::Project((*x_init)[i], jacobians[i]);

      if (!SO3Manifold::CheckTangentVector((*x_init)[i], TxU)) {
        std::cout << "CheckTangent False" << std::endl;
        return false;
      }
      std::cout << "TxU : " << TxU << std::endl;
      directions.push_back(TxU);
    }

    double step = BackTracing(cost_function, *x_init, directions);
    std::cout << "step size: " << step << std::endl;
    *x_init = New_X(*x_init, NewStep(directions, step));
    //SO3Manifold::TangentVector sk = -step * TxU;
    //(*x_init)[i] = SO3Manifold::Retraction((*x_init)[i], sk);

    std::cout << "error : " << cost_function.Evaluate(*x_init) << std::endl;
  }
  return true;
}


bool rgd(const std::shared_ptr<RGDFirstOrderInterface>& cost_function, Eigen::VectorXd* x_init) {
  size_t max_iteration = 32 * 32;
  size_t iteration = 0;

  //std::cout << "Initial error : " << cost_function->Evaluate(*x_init)
  //          << std::endl;
  while (iteration++ < max_iteration) {
    auto jacobians = cost_function->Jacobian(*x_init);

    Eigen::VectorXd TxU = cost_function->ProjectExtendedGradientToTangentSpace(
        *x_init, jacobians);
    //std::cout << "Iteration [" << iteration << "] TxU" << TxU << std::endl;
    double step = RGDBackTracking(cost_function, *x_init, -TxU);
    //std::cout << "Iteration ["<< iteration << "] step size: " << step << std::endl;
    *x_init = cost_function->Move(*x_init, step * -TxU);
    //std::cout << "Iteration [" << iteration
    //          << "] error : " << cost_function->Evaluate(*x_init) << std::endl;
  }
  return true;
}

double S(const Eigen::VectorXd& v, const Eigen::VectorXd& p, double radius) {
  double a = p.dot(p);
  double b = 2.0 * v.dot(p);
  double c = v.dot(v) - radius * radius;

  return (-b + std::sqrt(b * b - 4.0 * a * c)) / 2.0 / a;
}

Eigen::VectorXd ConjugateSolver(const Eigen::MatrixXd& H, const Eigen::VectorXd& b,
                     double radius = std::numeric_limits<double>::max()) {
  Eigen::VectorXd v = b;
  v.setZero();
  Eigen::VectorXd r = b;
  Eigen::VectorXd p = b;
  int iteration = 0;
  int max_iteration = 1024;
  while (iteration++ < max_iteration) {
    Eigen::VectorXd Hp = H * p;
    double last_r_norm = r.dot(r);
    double alpha = last_r_norm / p.dot(Hp);
    Eigen::VectorXd next_v = (v + alpha * p);

    if (next_v.norm() > radius) {
      double t = S(v, p, radius);
      std::cout << "Reach radius : "  << t << std::endl;
      return v + t * p;
    }

    v = (v  + alpha * p).eval();
    r = (r  - alpha * Hp).eval();
    if (r.norm() < 1e-3) {
      return v;
    }
    double beta = r.dot(r) / last_r_norm;
    p = r + beta * p;
  }

  if (iteration >= max_iteration) {
    std::cout << "Reach Max iteration" << std::endl;
  }
  return v;
}

bool AcceptOrNot(
    const std::shared_ptr<RiemannianSecondOrderInterface>& cost_function,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& gradient, const Eigen::MatrixXd& hess,
    const Eigen::VectorXd& step,
    double * radius
    ) {

      double accept_threshold = 0.1;
      double max_radius = 100;
      double n = -cost_function->Evaluate(cost_function->Move(x, step)) + cost_function->Evaluate(x);
      double d = -gradient.dot(step) - step.dot(hess * step);
      double rho = n / d;
      std::cout << "n : " << n << " d : " << d << " rho : " << rho << std::endl;

      bool res = rho > accept_threshold;

      if (rho < 0.25) {
        *radius *= 0.25;
      } else if (rho > 0.75 && step.dot(step) >= std::pow(*radius, 2)) {
        *radius = std::min(2 * *radius, max_radius);
      } else {
        *radius = *radius;
      }

      return res;
    }

bool RiemannianNewtonMethod(
    const std::shared_ptr<RiemannianSecondOrderInterface>& cost_function,
    Eigen::VectorXd* x_init) {
  size_t max_iteration = 32;
  size_t iteration = 0;
  double radius =  1.0;
  std::cout << "Initial error : " << cost_function->Evaluate(*x_init)
            << std::endl;
  while (iteration++ < max_iteration) {
    auto jacobians = cost_function->Jacobian(*x_init);
    Eigen::VectorXd TxU = cost_function->ProjectExtendedGradientToTangentSpace(
        *x_init, jacobians);
    Eigen::MatrixXd hess = cost_function->Hess(*x_init);
    Eigen::VectorXd step = ConjugateSolver(hess, -TxU, radius);
    std::cout << "Iteration ["<< iteration << "] step : " << step << std::endl;

    if (AcceptOrNot(cost_function, *x_init, TxU, hess, step, &radius)) {
      *x_init = cost_function->Move(*x_init, step);
    }
    std::cout << "Iteration [" << iteration
              << "] x : " << *x_init << " error : " << cost_function->Evaluate(*x_init) << std::endl;
    std::cout << "x norm : " << x_init->norm() << std::endl;
  }
  return true;
}

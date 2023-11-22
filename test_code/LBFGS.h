#ifndef OPTIMIZATION_LBFGS_H_
#define OPTIMIZATION_LBFGS_H_
#include <deque>
#include <stack>
template <int parameter_num>
Eigen::Matrix<double, parameter_num, 1> ComputeHessianInverseMultipleGradient(
    Eigen::Matrix<double, parameter_num, 1> &gradient,
    std::deque<Eigen::Matrix<double, parameter_num, 1>> &sk,
    std::deque<Eigen::Matrix<double, parameter_num, 1>> &yk) {
  // assume the H_0 is identity
  Eigen::Matrix<double, parameter_num, 1> r = gradient;
  assert(sk.size() == yk.size());
  std::stack<double> alphak;
 auto sk_rit = sk.rbegin();
 auto yk_rit = yk.rbegin();
  for (;
       sk_rit != sk.rend() && yk_rit != yk.rend(); ++sk_rit, ++yk_rit) {
    double demon = sk_rit->dot(*yk_rit) + 1e-9;
    double alpha = (*sk_rit).dot(r)  / demon;
    alphak.push(alpha);
    r = r - alpha * (*yk_rit) ;
  }
  auto sk_it = sk.begin();
  auto yk_it = yk.begin();
  for (;
       sk_it != sk.end() && yk_it != yk.end(); ++sk_it, ++yk_it) {
    double beta = (*yk_it).dot(r)  / (yk_it->dot(*sk_it) + 1e-9);
    r = r + (alphak.top() - beta) * (*sk_it);
    alphak.pop();
  }
  return r;
}
template <class Functor, int residual_num, int parameter_num>
bool LBFGS(AutoDiffFunction<Functor, residual_num, parameter_num> &functor,
           Eigen::Matrix<double, parameter_num, 1> &x0) {
  //Eigen::Matrix<double, parameter_num, parameter_num> hessian_inverse =
   //   Eigen::Matrix<double, parameter_num, parameter_num>::Identity();
  const double EPS = 1e-11;
  int max_num_iterator = 100;
  const size_t max_history_num = 2;
  std::deque<Eigen::Matrix<double, parameter_num, 1>> sk, yk;
  // H_inverse_0 is Identity;
  for (int iterator = 0; iterator < max_num_iterator; iterator++) {
    //auto error = functor(x0);
    auto gradient = functor.Jacobian(x0);
    if (gradient.norm() < EPS) {
      return true;
    }

    Eigen::Matrix<double, parameter_num, 1> direct =
        ComputeHessianInverseMultipleGradient(gradient, sk, yk);
        direct = -direct;
    double t = BackTracing(functor, x0, direct);
    x0 = x0 + t * direct;
    DLOG(INFO) << "Move : " << t * direct.transpose() << std::endl;
    DLOG(INFO) << "x : " << x0.transpose() << std::endl;
    Eigen::Matrix<double, parameter_num, 1> s = t * direct;
    auto new_gradient = functor.Jacobian(x0);
    Eigen::Matrix<double, parameter_num, 1> y = new_gradient - gradient;
    if (sk.size() == max_history_num) {
      sk.pop_front();
    }
    sk.push_back(s);
    if (yk.size() == max_history_num) {
      yk.pop_front();
    }
    yk.push_back(y);
  }
  return true;
}
#endif  // OPTIMIZATION_LBFGS_H_

#ifndef OPTIMIZATION_JET_H
#define OPTIMIZATION_JET_H
#include <Eigen/Dense>
template <int N>
class Jet {
 private:
  double value_;
  Eigen::Matrix<double, N, 1> gradient_;

 public:
  Jet(double value = 0.0) : value_(value), gradient_() {}
  Jet(double value, int index) : value_(value), gradient_() {
    gradient_.Zero();
    gradient_(index) = 1.0;
  }

  Jet(double value, Eigen::Matrix<double, N, 1> gradient)
      : value_(value), gradient_(gradient) {}
  double value() const { return value_; }

  Eigen::Matrix<double, N, 1> Gradient() { return gradient_; }

  template <int T>
  friend Jet<T> operator+(const Jet<T>& lhs, const Jet<T>& rhs);
  template <int T>
  friend Jet<T> operator-(const Jet<T>& lhs, const Jet<T>& rhs);
  template <int T>
  friend Jet<T> operator*(const Jet<T>& lhs, const Jet<T>& rhs);
  template <int T>
  friend Jet<T> operator/(const Jet<T>& lhs, const Jet<T>& rhs);
};

template <int N>
Jet<N> operator+(const Jet<N>& lhs, const Jet<N>& rhs) {
  return Jet<N>(lhs.value_ + rhs.value_, lhs.gradient_ + rhs.gradient_);
}
template <int N>
Jet<N> operator-(const Jet<N>& lhs, const Jet<N>& rhs) {
  return Jet<N>(lhs.value_ - rhs.value_, lhs.gradient_ - rhs.gradient_);
}
template <int N>
Jet<N> operator*(const Jet<N>& lhs, const Jet<N>& rhs) {
  return Jet<N>(lhs.value_ * rhs.value_,
                lhs.gradient_ * rhs.value_ + rhs.gradient_ * lhs.value_);
}
template <int N>
Jet<N> operator/(const Jet<N>& lhs, const Jet<N>& rhs) {
  return Jet<N>(lhs.value_ * rhs.value_,
                lhs.gradient_ / rhs.value -
                    rhs.gradient_ * lhs.value_ / rhs.value / rhs.value);
}

template <class Functor, int residual_num, int parameter_num>
class AutoDiffFunction {
 public:
  AutoDiffFunction(Functor&& functor)
      : functor_(std::forward<Functor>(functor)) {}

  Eigen::Matrix<double, residual_num, 1> operator()(
      const Eigen::Matrix<double, parameter_num, 1>& x) const {
     std::vector<Jet<parameter_num>> x_wrap(parameter_num);
    for (size_t i = 0; i < parameter_num; i++) {
      x_wrap[i] = Jet<parameter_num>(x(i), i);
    }
    std::vector<Jet<parameter_num>> residual_wrap(residual_num);
    functor_(x_wrap.data(), residual_wrap.data());
    Eigen::Matrix<double, residual_num, 1> res;
    for (int i = 0; i < residual_num; i++) {
      res(i) = residual_wrap[i].value();
    }
    return res;
  }
Eigen::Matrix<double, residual_num, parameter_num> Gradient(
      Eigen::Matrix<double, parameter_num, 1>& x) const {
     std::vector<Jet<parameter_num>> x_wrap(parameter_num);
    for (size_t i = 0; i < parameter_num; i++) {
      x_wrap[i] = Jet<parameter_num>(x(i), i);
    }
    std::vector<Jet<parameter_num>> residual_wrap(residual_num);
    functor_(x_wrap.data(), residual_wrap.data());
    Eigen::Matrix<double, residual_num, parameter_num> res;
    for (int i = 0; i < residual_num; i++) {
        for (int j = 0; j < parameter_num; j++) {
            res(i, j) = residual_wrap[i].Gradient()(j);
        }
    }
    return res;
  }
  Eigen::Matrix<double, parameter_num, residual_num> Jacobian(
    Eigen::Matrix<double, parameter_num, 1>& x
  ) const {
      return Gradient(x).transpose();
  }
 private:
  Functor functor_;
};
#endif  // OPTIMIZATION_JET_H
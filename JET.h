#ifndef OPTIMIZATION_JET_H
#define OPTIMIZATION_JET_H
#include <Eigen/Dense>
template <int N>
class Jet {
 private:
  double value_;
  Eigen::Matrix<double, N, 1> gradient_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Jet(double value = 0.0) : value_(value), gradient_() {
    for (int i = 0; i < N; i++) {
      gradient_(i) = 0.0;
    }
  }
  Jet(double value, int index) : value_(value), gradient_() {
    for (int i = 0; i < N; i++) {
      gradient_(i) = 0.0;
    }
    gradient_(index) = 1.0;
  }

  Jet(double value, Eigen::Matrix<double, N, 1> gradient)
      : value_(value), gradient_(gradient) {}
  double value() const { return value_; }

  Eigen::Matrix<double, N, 1> Gradient() const { return gradient_; }

  template <int T>
  friend Jet<T> operator+(const Jet<T>& lhs, const Jet<T>& rhs);
  template <int T>
  friend Jet<T> operator+(const Jet<T>& lhs, double s);
  template <int T>
  friend Jet<T> operator+(double s, const Jet<T>& lhs);

  template <int T>
  friend Jet<T> operator-(const Jet<T>& lhs, const Jet<T>& rhs);
  template <int T>
  friend Jet<T> operator-(const Jet<T>& lhs, double s);
  template <int T>
  friend Jet<T> operator-(double s, const Jet<T>& lhs);

  template <int T>
  friend Jet<T> operator*(const Jet<T>& lhs, const Jet<T>& rhs);
  template <int T>
  friend Jet<T> operator*(const Jet<T>& lhs, double s);
  template <int T>
  friend Jet<T> operator*(double s, const Jet<T>& lhs);

  template <int T>
  friend Jet<T> operator/(const Jet<T>& lhs, const Jet<T>& rhs);
  template <int T>
  friend Jet<T> operator/(const Jet<T>& lhs, double s);
  template <int T>
  friend Jet<T> operator/(double s, const Jet<T>& lhs);

  template <int T>
  friend Jet<T> sin(const Jet<T>& sour);
  template <int T>
  friend Jet<T> cos(const Jet<T>& sour);

  template <int T>
  friend Jet<T> sqrt(const Jet<T>& sour);
};

template <int N>
Jet<N> operator+(const Jet<N>& lhs, const Jet<N>& rhs) {
  return Jet<N>(lhs.value_ + rhs.value_, lhs.gradient_ + rhs.gradient_);
}

template <int N>
Jet<N> operator+(const Jet<N>& lhs, double s) {
  return Jet<N>(lhs.value_ + s, lhs.gradient_);
}

template <int N>
Jet<N> operator+(double s, const Jet<N>& lhs) {
  return Jet<N>(lhs.value_ + s, lhs.gradient_);
}

template <int N>
Jet<N> operator-(const Jet<N>& lhs, const Jet<N>& rhs) {
  return Jet<N>(lhs.value_ - rhs.value_, lhs.gradient_ - rhs.gradient_);
}

template <int N>
Jet<N> operator-(const Jet<N>& lhs, double s) {
  return Jet<N>(lhs.value_ - s, lhs.gradient_);
}

template <int N>
Jet<N> operator-(double s, const Jet<N>& lhs) {
  return Jet<N>(s - lhs.value_, lhs.gradient_);
}
template <int N>
Jet<N> operator*(const Jet<N>& lhs, const Jet<N>& rhs) {
  return Jet<N>(lhs.value_ * rhs.value_,
                lhs.gradient_ * rhs.value_ + rhs.gradient_ * lhs.value_);
}

template <int N>
Jet<N> operator*(const Jet<N>& lhs, double s) {
  return Jet<N>(lhs.value_ * s, s * lhs.gradient_);
}

template <int N>
Jet<N> operator*(double s, const Jet<N>& lhs) {
  return Jet<N>(lhs.value_ * s, s * lhs.gradient_);
}

template <int N>
Jet<N> operator/(const Jet<N>& lhs, const Jet<N>& rhs) {
  return Jet<N>(lhs.value_ * rhs.value_,
                lhs.gradient_ / rhs.value_ -
                    rhs.gradient_ * lhs.value_ / rhs.value_ / rhs.value_);
}

template <int N>
Jet<N> operator/(const Jet<N>& lhs, double s) {
  return Jet<N>(lhs.value_ / s, lhs.gradient_ / s);
}

template <int N>
Jet<N> operator/(double s, const Jet<N>& lhs) {
  const double minus_s_g_a_inverse2 = -s / (lhs.gradient_ * lhs.gradient_);
  return Jet<N>(s / lhs.value_, lhs.gradient_ * minus_s_g_a_inverse2);
}

template <int N>
Jet<N> sqrt(const Jet<N>& sour) {
  const double tmp = sqrt(sour.value_);
  const double two_a_inverse = double(1.0) / (double(2.0) * tmp);
  return Jet<N>(tmp, sour.gradient_ * two_a_inverse);
}
template <int N>
Jet<N> sin(const Jet<N>& sour) {
  return Jet<N>(std::sin(sour.value_), std::cos(sour.value_) * sour.gradient_);
}

template <int N>
Jet<N> cos(const Jet<N>& sour) {
  return Jet<N>(std::cos(sour.value_), -std::sin(sour.value_) * sour.gradient_);
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
    functor_(&x_wrap[0], &residual_wrap[0]);
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
    functor_(&x_wrap[0], &residual_wrap[0]);
    Eigen::Matrix<double, residual_num, parameter_num> res;
    for (int i = 0; i < residual_num; i++) {
      Eigen::Matrix<double, parameter_num, 1> gradient =
          residual_wrap[i].Gradient();
      for (int j = 0; j < parameter_num; j++) {
        res(i, j) = gradient(j);
      }
    }
    return res;
  }
  Eigen::Matrix<double, parameter_num, residual_num> Jacobian(
      Eigen::Matrix<double, parameter_num, 1>& x) const {
    return Gradient(x).transpose();
  }

 private:
  Functor functor_;
};
#endif  // OPTIMIZATION_JET_H
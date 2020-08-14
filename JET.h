#ifndef OPTIMIZATION_JET_H
#define OPTIMIZATION_JET_H
#include <Eigen/Dense>
template<int N>
class Jet {
private:
double value_;
Eigen::Matrix<double, N, 1> gradient_;
public:
Jet(double value, int index) : value_(value), gradient_() {
    gradient_.Zero();
    gradient_(index) = 1.0;
}

Jet(double value, Eigen::Matrix<double, N, 1> gradient) : value_(value), gradient_(gradient) {}

const Eigen::Matrix<double, N, 1> Gradient() { return gradient_;}
template<int T>
friend Jet<T> operator+(const Jet<T>& lhs, const Jet<T>& rhs);
};

template<int N>
Jet<N> operator+(const Jet<N>& lhs, const Jet<N>& rhs) {
    return Jet<N>(lhs.value_ + rhs.value_, lhs.gradient_ + rhs.gradient_);
}

#endif  // OPTIMIZATION_JET_H
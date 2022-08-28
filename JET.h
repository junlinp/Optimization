#ifndef OPTIMIZATION_JET_H
#define OPTIMIZATION_JET_H
#include <assert.h>

#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>

#include "Eigen/Dense"

// CRTP
template <class Derive, class Data_Type, int Dimension> class Expression {
public:
  using DATA_TYPE = Data_Type;
  static const int DUAL_NUMBER_SIZE = Dimension;
  using GRADIENT_TYPE = Eigen::Matrix<DATA_TYPE, DUAL_NUMBER_SIZE, 1>;

  DATA_TYPE value() const { return impl().value_imp(); }
  GRADIENT_TYPE Gradient() const { return impl().Gradient_imp(); }

private:
  Derive &impl() { return *static_cast<Derive *>(this); }
  const Derive &impl() const { return *static_cast<const Derive *>(this); }
};

template <class Type,
          typename U = typename std::remove_reference_t<Type>::DATA_TYPE>
struct IMPL_JET_Concept_ {
  using type = bool;
};

template <class Type>
using JET_Concept = typename IMPL_JET_Concept_<Type>::type;
// BASIC_TYPE should be float or double
// and there is a problem to initialize a array of Jet.
template <class BASIC_TYPE, int N>
class Jet : public Expression<Jet<BASIC_TYPE, N>, BASIC_TYPE, N> {
public:
  using DATA_TYPE = BASIC_TYPE;
  static const int DUAL_NUMBER_SIZE = N;
  using GRADIENT_TYPE = Eigen::Matrix<DATA_TYPE, DUAL_NUMBER_SIZE, 1>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Jet(BASIC_TYPE value = 0.0) : value_(value), gradient_() {
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

  template <class EXPR, JET_Concept<EXPR> = true>
  Jet &operator=(const EXPR &expr) {
    value_ = expr.value();
    gradient_ = expr.Gradient();
    return *this;
  }

  template <class EXPR, JET_Concept<EXPR> = true>
  Jet(const EXPR &expr) : value_{expr.value()}, gradient_(expr.Gradient()) {
  }

  template <class EXPR, JET_Concept<EXPR> = true>
  Jet(EXPR &&expr) : value_(expr.value()), gradient_(expr.Gradient()) {
  }

  template <class EXPR, typename DataType = typename EXPR::DATA_TYPE>
  Jet &operator+=(const EXPR &expr) {
    value_ += expr.value();
    gradient_ += expr.Gradient();
    return *this;
  }

  template <
      class DOUBLE_POINT,
      std::enable_if_t<std::is_floating_point_v<DOUBLE_POINT>, bool> = true>
  Jet &operator+=(const DOUBLE_POINT &number) {
    return *this += Jet(number);
  }

  template <class EXPR, typename DataType = typename EXPR::DATA_TYPE>
  Jet &operator-=(const EXPR &expr) {
    value_ -= expr.value();
    gradient_ -= expr.Gradient();
    return *this;
  }

  template <
      class DOUBLE_POINT,
      std::enable_if_t<std::is_floating_point_v<DOUBLE_POINT>, bool> = true>
  Jet &operator-=(const DOUBLE_POINT &number) {
    return *this -= Jet(number);
  }

  template <class EXPR> Jet &operator/=(const EXPR &expr) {
    value_ /= expr.value();
    gradient_ = (value_ * expr.Gradient() - expr.value() * gradient_) /
                (expr.value() * expr.value());
    return *this;
  }

  void setGradient(size_t index, BASIC_TYPE value) { gradient_(index) = value; }
  DATA_TYPE value_imp() const { return value_; }

  GRADIENT_TYPE Gradient_imp() const { return gradient_; }

private:
  BASIC_TYPE value_;
  GRADIENT_TYPE gradient_;
};

template <int N> using JETF = Jet<float, N>;

template <int N> using JETD = Jet<double, N>;

template <class OPRAND, class OperatorImp,
          typename RM_OPRAND = std::remove_reference_t<OPRAND>>
class UnaryOp : public Expression<UnaryOp<OPRAND, OperatorImp>,
                                  typename RM_OPRAND::DATA_TYPE,
                                  RM_OPRAND::DUAL_NUMBER_SIZE> {
public:
  using DATA_TYPE = typename RM_OPRAND::DATA_TYPE;
  static constexpr int DUAL_NUMBER_SIZE = RM_OPRAND::DUAL_NUMBER_SIZE;
  using GRADIENT_TYPE = typename RM_OPRAND::GRADIENT_TYPE;

  UnaryOp(OPRAND &&oprand) : oprand_{std::forward<OPRAND>(oprand)} {}

  DATA_TYPE value_imp() const { return OperatorImp::value_unary_op(oprand_); }

  GRADIENT_TYPE Gradient_imp() const {
    return OperatorImp::Gradient_unary_op(oprand_);
  }

private:
  OPRAND oprand_;
};

class MinusUnaryOp {
public:
  template <typename EXPR>
  static typename EXPR::DATA_TYPE value_unary_op(const EXPR &expr) {
    return -expr.value();
  }

  template <typename EXPR>
  static typename EXPR::GRADIENT_TYPE Gradient_unary_op(const EXPR &expr) {
    return -expr.Gradient();
  }
};

template <typename EXPR, JET_Concept<EXPR> = true>
auto operator-(EXPR &&oprand) {
  return UnaryOp<EXPR, MinusUnaryOp>(std::forward<EXPR>(oprand));
}

class SinOp {
public:
  template <typename EXPR> static auto value_unary_op(const EXPR &expr) {
    return sin(expr.value());
  }
  template <typename EXPR> static auto Gradient_unary_op(const EXPR &expr) {
    using GRADIENT_TYPE = std::remove_cv_t<typename EXPR::GRADIENT_TYPE>;
    GRADIENT_TYPE gradient = expr.Gradient();
    for (int i = 0; i < 12; i++) {
      gradient(i) = std::cos(gradient(i));
    }
    return gradient;
  }
};

template <typename EXPR, JET_Concept<EXPR> = true> auto sin(EXPR &&oprand) {
  return UnaryOp<EXPR, SinOp>(std::forward<EXPR>(oprand));
}

class CosOp {
public:
  template <typename EXPR>
  static typename EXPR::DATA_TYPE value_unary_op(const EXPR &expr) {
    return cos(expr.value());
  }
  template <typename EXPR>
  static typename EXPR::GRADIENT_TYPE Gradient_unary_op(const EXPR &expr) {
    using GRADIENT_TYPE = std::remove_cv_t<typename EXPR::GRADIENT_TYPE>;
    GRADIENT_TYPE gradient = expr.Gradient();
    for (int i = 0; i < EXPR::DUAL_NUMBER_SIZE; i++) {
      gradient(i) = -std::sin(gradient(i));
    }
    return gradient;
  }
};

template <typename EXPR, JET_Concept<EXPR> = true> auto cos(EXPR &&oprand) {
  return UnaryOp<EXPR, CosOp>(std::forward<EXPR>(oprand));
}

class SqrtOp {
public:
  template <typename EXPR> static auto value_unary_op(const EXPR &expr) {
    return std::sqrt(expr.value());
  }
  template <typename EXPR> static auto Gradient_unary_op(const EXPR &expr) {
    return (expr.Gradient() * 0.5 / std::sqrt(expr.value())).eval();
  }
};
template <typename EXPR, JET_Concept<EXPR> = true> auto sqrt(EXPR &&oprand) {
  return UnaryOp<EXPR, SqrtOp>(std::forward<EXPR>(oprand));
}

template <class LHS_OPRAND, class RHS_OPRAND> struct Binary_trait {
  static_assert(
      std::is_same<typename LHS_OPRAND::DATA_TYPE,
                   typename RHS_OPRAND::DATA_TYPE>::value,
      "The Data Type between LHS Oprand and RHS Oprand is different.");
  static_assert(
      LHS_OPRAND::DUAL_NUMBER_SIZE == RHS_OPRAND::DUAL_NUMBER_SIZE,
      "The Dual Number Size between LHS Oprand and RHS Oprand is different.");

  using DATA_TYPE = typename LHS_OPRAND::DATA_TYPE;
  const static int DUAL_NUMBER_SIZE = LHS_OPRAND::DUAL_NUMBER_SIZE;
  using GRADIENT_TYPE = typename LHS_OPRAND::GRADIENT_TYPE;
};

template <class LHS_OPRAND, class RHS_OPRAND, class OperatorImp,
          typename CVRM_LHS_OPRAND = std::remove_reference_t<LHS_OPRAND>,
          typename CVRM_RHS_OPRAND = std::remove_reference_t<RHS_OPRAND>>
class BinaryOp
    : public Expression<
          BinaryOp<LHS_OPRAND, RHS_OPRAND, OperatorImp>,
          typename Binary_trait<CVRM_LHS_OPRAND, CVRM_RHS_OPRAND>::DATA_TYPE,
          Binary_trait<CVRM_LHS_OPRAND, CVRM_RHS_OPRAND>::DUAL_NUMBER_SIZE> {
public:
  using DATA_TYPE =
      typename Binary_trait<CVRM_LHS_OPRAND, CVRM_RHS_OPRAND>::DATA_TYPE;
  static constexpr int DUAL_NUM_SIZE =
      Binary_trait<CVRM_LHS_OPRAND, CVRM_RHS_OPRAND>::DUAL_NUMBER_SIZE;
  using GRADIENT_TYPE =
      typename Binary_trait<CVRM_LHS_OPRAND, CVRM_RHS_OPRAND>::GRADIENT_TYPE;

  BinaryOp(LHS_OPRAND &&lhs, RHS_OPRAND &&rhs)
      : lhs_oprand_{std::forward<LHS_OPRAND>(lhs)},
        rhs_oprand_{std::forward<RHS_OPRAND>(rhs)} {
  }

  BinaryOp(BinaryOp &&other)
      : lhs_oprand_(std::forward<decltype(lhs_oprand_)>(other.lhs_oprand_)),
        rhs_oprand_(std::forward<decltype(rhs_oprand_)>(other.rhs_oprand_)) {
  }

  DATA_TYPE value_imp() const {
    return OperatorImp::value_binary_op(lhs_oprand_, rhs_oprand_);
  }

  Eigen::Matrix<double, DUAL_NUM_SIZE, 1> Gradient_imp() const {
    return OperatorImp::Gradient_binary_op(lhs_oprand_, rhs_oprand_);
  }

private:
  BinaryOp(const BinaryOp &other)
      : lhs_oprand_(other.lhs_oprand_), rhs_oprand_(other.rhs_oprand_) {}
  BinaryOp &operator=(const BinaryOp &);

  LHS_OPRAND lhs_oprand_;
  RHS_OPRAND rhs_oprand_;
};

class PlusOp {
public:
  template <typename LHS_OPRAND, typename RHS_OPRAND>
  static auto value_binary_op(const LHS_OPRAND &lhs, const RHS_OPRAND &rhs) {
    return lhs.value() + rhs.value();
  }

  template <typename LHS_OPRAND, typename RHS_OPRAND>
  static auto Gradient_binary_op(const LHS_OPRAND &lhs, const RHS_OPRAND &rhs) {
    return (lhs.Gradient() + rhs.Gradient()).eval();
  }
};

// SFINAE
template <class LHS, class RHS, JET_Concept<LHS> = true,
          JET_Concept<RHS> = true>
auto operator+(LHS &&lhs, RHS &&rhs) {
  return BinaryOp<LHS, RHS, PlusOp>(std::forward<LHS>(lhs),
                                    std::forward<RHS>(rhs));
}

template <class LHS, class RHS, JET_Concept<LHS> = true,
          std::enable_if_t<std::is_floating_point_v<RHS>, bool> = true>
auto operator+(LHS &&lhs, RHS &&rhs) {
  using DATA_TYPE = typename std::remove_reference_t<LHS>::DATA_TYPE;
  constexpr int DUAL_NUMBER_SIZE =
      std::remove_reference_t<LHS>::DUAL_NUMBER_SIZE;
  using RHS_J = Jet<DATA_TYPE, DUAL_NUMBER_SIZE>;
  static_assert(std::is_same_v<RHS_J, JETD<1>>, "null");
  RHS_J rhs_{rhs};
  return BinaryOp<LHS, RHS_J, PlusOp>(std::forward<LHS>(lhs),
                                          std::move(rhs_));
}

class MinusBinaryOp {
public:
  template <class LHS_OPRAND, class RHS_OPRAND>
  static auto value_binary_op(const LHS_OPRAND &lhs, const RHS_OPRAND &rhs) {
    return lhs.value() - rhs.value();
  }
  template <class LHS_OPRAND, class RHS_OPRAND>
  static auto Gradient_binary_op(const LHS_OPRAND &lhs, const RHS_OPRAND &rhs) {
    return (lhs.Gradient() - rhs.Gradient()).eval();
  }
};
template <class LHS, class RHS, JET_Concept<LHS> = true,
          JET_Concept<RHS> = true>
auto operator-(LHS &&lhs, RHS &&rhs) {
  return BinaryOp<LHS, RHS, MinusBinaryOp>(std::forward<LHS>(lhs),
                                           std::forward<RHS>(rhs));
}

class MultipleOp {
public:
  template <class LHS_OPRAND, class RHS_OPRAND>
  static auto value_binary_op(const LHS_OPRAND &lhs, const RHS_OPRAND &rhs) {
    return lhs.value() * rhs.value();
  }
  template <class LHS_OPRAND, class RHS_OPRAND>
  static auto Gradient_binary_op(const LHS_OPRAND &lhs, const RHS_OPRAND &rhs) {
    static_assert(std::is_same_v<typename LHS_OPRAND::GRADIENT_TYPE,
                                 typename RHS_OPRAND::GRADIENT_TYPE>,
                  "Gradient Data");
    using Gradient_Type = typename LHS_OPRAND::GRADIENT_TYPE;
    return Gradient_Type{
        (rhs.value() * lhs.Gradient() + lhs.value() * rhs.Gradient()).eval()};
  }
};

template <class LHS, class RHS, JET_Concept<LHS> = true,
          JET_Concept<RHS> = true>
auto operator*(LHS &&lhs, RHS &&rhs) {
  return BinaryOp<LHS, RHS, MultipleOp>(std::forward<LHS>(lhs),
                                        std::forward<RHS>(rhs));
}

class DivisionOp {
public:
  template <class LHS_OPRAND, class RHS_OPRAND>
  static auto value_binary_op(const LHS_OPRAND &lhs, const RHS_OPRAND &rhs) {
    return lhs.value() /
           (rhs.value() +
            std::numeric_limits<typename RHS_OPRAND::DATA_TYPE>::epsilon());
  }

  template <class LHS_OPRAND, class RHS_OPRAND>
  static auto Gradient_binary_op(const LHS_OPRAND &lhs, const RHS_OPRAND &rhs) {
    // f / g = f' * g - g' * f / g / g
    static_assert(std::is_same_v<typename LHS_OPRAND::GRADIENT_TYPE,
                                 typename RHS_OPRAND::GRADIENT_TYPE>,
                  "Gradient Data");
    auto &&l = lhs.value();
    auto &&r = rhs.value();
    typename RHS_OPRAND::DATA_TYPE denominator =
        (r * r +
         std::numeric_limits<typename RHS_OPRAND::DATA_TYPE>::epsilon());
    return ((r * lhs.Gradient() - l * rhs.Gradient()) / denominator).eval();
  }
};

template <class LHS, class RHS, JET_Concept<LHS> = true,
          JET_Concept<RHS> = true>
auto operator/(LHS &&lhs, RHS &&rhs) {
  return BinaryOp<LHS, RHS, DivisionOp>(std::forward<LHS>(lhs),
                                        std::forward<RHS>(rhs));
}

template <class LHS, class RHS, JET_Concept<LHS> = true,
          std::enable_if_t<std::is_floating_point_v<RHS>, bool> = true>
auto operator/(LHS &&lhs, RHS &&rhs) {
  using DATA_TYPE = typename std::remove_reference_t<LHS>::DATA_TYPE;
  constexpr int DUAL_NUMBER_SIZE =
      std::remove_reference_t<LHS>::DUAL_NUMBER_SIZE;
  using RHS_J = Jet<DATA_TYPE, DUAL_NUMBER_SIZE>;
  static_assert(std::is_same_v<RHS_J, JETD<1>>, "null");
  RHS_J rhs_{rhs};
  return BinaryOp<LHS, RHS_J, DivisionOp>(std::forward<LHS>(lhs),
                                          std::move(rhs_));
}

template <int residual_num, int parameter_num, class Functor, class X>
bool GradientCheck(Functor&& functor,X* x, double EPSILON) {
  JETD<parameter_num> x_origin[parameter_num], y_origin[parameter_num];
  for(int parameter_index = 0; parameter_index < parameter_num; parameter_index++) {
    x_origin[parameter_index] = JETD<parameter_num>{x[parameter_index] + EPSILON, parameter_index};
  }
  functor(x_origin, y_origin);

  for(int residual_index = 0; residual_index < residual_num; residual_index++) {
    JETD<parameter_num> x_plus[parameter_num], x_sub[parameter_num];
    JETD<parameter_num> y_plus[parameter_num], y_sub[parameter_num];
    for(int parameter_index = 0; parameter_index < parameter_num; parameter_index++) {
      int check_index = parameter_index;
      std::memcpy(x_plus, x_origin, parameter_num * sizeof(JETD<parameter_num>));
      std::memcpy(x_sub, x_origin, parameter_num * sizeof(JETD<parameter_num>));

      x_plus[check_index] = JETD<parameter_num>{x[check_index] + EPSILON, check_index};
      x_sub[check_index] = JETD<parameter_num>{x[check_index] - EPSILON, check_index};
      functor(x_plus, y_plus);
      functor(x_sub, y_sub);

      double gradient = (y_plus[residual_index].value() - y_sub[residual_index].value()) * 0.5 / EPSILON;

      if ( std::abs(gradient - y_origin[residual_index].Gradient()(check_index)) > EPSILON) {
        return false;
      }
    }
  }
  return true;
}

template <class Functor, int residual_num, int parameter_num>
class AutoDiffFunction {
public:
  AutoDiffFunction(Functor &&functor)
      : functor_(std::forward<Functor>(functor)) {}

  Eigen::Matrix<double, residual_num, 1>
  operator()(const Eigen::Matrix<double, parameter_num, 1> &x) const {
    std::vector<JETD<parameter_num>> x_wrap(parameter_num);
    for (size_t i = 0; i < parameter_num; i++) {
      x_wrap[i] = JETD<parameter_num>(x(i), i);
    }
    std::vector<JETD<parameter_num>> residual_wrap(residual_num);
    functor_(&x_wrap[0], &residual_wrap[0]);
    Eigen::Matrix<double, residual_num, 1> res;
    for (int i = 0; i < residual_num; i++) {
      res(i) = residual_wrap[i].value();
    }
    return res;
  }
  Eigen::Matrix<double, residual_num, parameter_num>
  Gradient(Eigen::Matrix<double, parameter_num, 1> &x) const {
    std::vector<JETD<parameter_num>> x_wrap(parameter_num);

    for (size_t i = 0; i < parameter_num; i++) {
      x_wrap[i] = JETD<parameter_num>(x(i), i);
    }
    std::vector<JETD<parameter_num>> residual_wrap(residual_num);

    functor_(x_wrap.data(), residual_wrap.data());

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
  Eigen::Matrix<double, parameter_num, residual_num>
  Jacobian(Eigen::Matrix<double, parameter_num, 1> &x) const {
    return Gradient(x).transpose();
  }

private:
  Functor functor_;
};
#endif // OPTIMIZATION_JET_H

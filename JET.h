#ifndef OPTIMIZATION_JET_H
#define OPTIMIZATION_JET_H
#include <assert.h>

#include <Eigen/Dense>
#include <type_traits>

// CRTP
template <class Derive, class Data_Type, int Dimension>
class Expression {
 public:
  using DATA_TYPE = Data_Type;
  static const int DUAL_NUMBER_SIZE = Dimension;
  using GRADIENT_TYPE = Eigen::Matrix<DATA_TYPE, DUAL_NUMBER_SIZE, 1>;

  DATA_TYPE value() const { return impl().value_imp(); }
  GRADIENT_TYPE Gradient() const { return impl().Gradient_imp(); }

 private:
  Derive& impl() { return *static_cast<Derive*>(this); }
  const Derive& impl() const { return *static_cast<const Derive*>(this); }
};

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
  template <class EXPR>
  Jet& operator=(const EXPR& expr) {
    value_ = expr.value();
    gradient_ = expr.Gradient();
    return *this;
  }

  template <class EXPR>
  Jet(const EXPR& expr) : value_{expr.value()}, gradient_(expr.Gradient()) {}
  
  template <class EXPR>
  Jet& operator+=(const EXPR& expr) {
    value_ += expr.value();
    gradient_ += expr.Gradient();
    return *this;
  }

  template<class EXPR>
  Jet& operator/=(const EXPR& expr) {
    value_ /= expr.value();
    gradient_ = (value_ * expr.Gradient() - expr.value() * gradient_) / (expr.value() * expr.value());
    return *this;
  }

  void setGradient(size_t index, BASIC_TYPE value) { gradient_(index) = value; }
  DATA_TYPE value_imp() const { return value_; }

  GRADIENT_TYPE Gradient_imp() const { return gradient_; }

 private:
  BASIC_TYPE value_;
  GRADIENT_TYPE gradient_;
};

template <int N>
using JETF = Jet<float, N>;

template <int N>
using JETD = Jet<double, N>;

template <class OPRAND, class OperatorImp>
class UnaryOp
    : public Expression<UnaryOp<OPRAND, OperatorImp>, typename OPRAND::DATA_TYPE,
                        OPRAND::DUAL_NUMBER_SIZE> {
 public:
  using DATA_TYPE = typename OPRAND::DATA_TYPE;
  static const int DUAL_NUMBER_SIZE = OPRAND::DUAL_NUMBER_SIZE;
  UnaryOp(const OPRAND& oprand) : oprand_(oprand) {}

  DATA_TYPE value_imp() const {
    return OperatorImp::value_unary_op(oprand_);
  }
  Eigen::Matrix<DATA_TYPE, DUAL_NUMBER_SIZE, 1> Gradient_imp() {
    return OperatorImp::Gradient_unary_op(oprand_);
  }

 private:
  const OPRAND& oprand_;
};

class MinusUnaryOp {
 public:
 template<typename EXPR>
  static auto value_unary_op(const EXPR& expr) { return -expr.value(); }
  template<typename EXPR>
  static auto Gradient_unary_op(
      const EXPR& expr) {
    return -expr.Gradient();
  }
};

class SinOp {
  public:
  template<typename EXPR>
  static auto value_unary_op(const EXPR& expr) { return sin(expr.value());}
  template<typename EXPR>
  static auto Gradient_unary_op(const EXPR& expr) {
    return std::cos(expr.Gradient());
  }
};

class CosOp {
  public:
  template<typename EXPR>
  static auto value_unary_op(const EXPR& expr) { return cos(expr.value());}
  template<typename EXPR>
  static auto Gradient_unary_op(const EXPR& expr) {
    return -std::sin(expr.Gradient());
  }
};
template <typename EXPR, class DataType = typename EXPR::DATA_TYPE>
auto cos(const EXPR& oprand) {
  return UnaryOp<EXPR, CosOp>(oprand);
}

class SqrtOp {
  public:
  template<typename EXPR>
  static auto value_unary_op(const EXPR& expr) { return sqrt(expr.value());}
  template<typename EXPR>
  static auto Gradient_unary_op(const EXPR& expr) {
    return expr.Gradient() / sqrt(expr.value());
  }
};
template <typename EXPR, class DataType = typename EXPR::DATA_TYPE>
auto sqrt(const EXPR& oprand) {
  return UnaryOp<EXPR, SqrtOp>(oprand);
}


template <typename EXPR, class DataType = typename EXPR::DATA_TYPE>
auto operator-(const EXPR& oprand) {
  return UnaryOp<EXPR, MinusUnaryOp>(oprand);
}

template <typename EXPR, class DataType = typename EXPR::DATA_TYPE>
auto sin(const EXPR& oprand) {
  return UnaryOp<EXPR, SinOp>(oprand);
}

template <class LHS_OPRAND, class RHS_OPRAND>
struct Binary_trait {
  static_assert(
      std::is_same<typename LHS_OPRAND::DATA_TYPE,
                   typename RHS_OPRAND::DATA_TYPE>::value,
      "The Data Type between LHS Oprand and RHS Oprand is different.");
  static_assert(
      LHS_OPRAND::DUAL_NUMBER_SIZE == RHS_OPRAND::DUAL_NUMBER_SIZE,
      "The Dual Number Size between LHS Oprand and RHS Oprand is different.");

  using DATA_TYPE = typename LHS_OPRAND::DATA_TYPE;
  const static int DUAL_NUMBER_SIZE = LHS_OPRAND::DUAL_NUMBER_SIZE;
};

template <class LHS_OPRAND, class RHS_OPRAND, class OperatorImp>
class BinaryOp : public Expression<
                     BinaryOp<LHS_OPRAND, RHS_OPRAND, OperatorImp>,
                     typename Binary_trait<LHS_OPRAND, RHS_OPRAND>::DATA_TYPE,
                     Binary_trait<LHS_OPRAND, RHS_OPRAND>::DUAL_NUMBER_SIZE> {
 public:
  using DATA_TYPE = typename Binary_trait<LHS_OPRAND, RHS_OPRAND>::DATA_TYPE;
  static const int DUAL_NUM_SIZE =
      Binary_trait<LHS_OPRAND, RHS_OPRAND>::DUAL_NUMBER_SIZE;
  BinaryOp(const LHS_OPRAND& lhs, const RHS_OPRAND& rhs)
      : lhs_oprand_(lhs), rhs_oprand_(rhs) {}

  DATA_TYPE value_imp() const {
    return OperatorImp::value_binary_op(lhs_oprand_, rhs_oprand_);
  }

  Eigen::Matrix<double, DUAL_NUM_SIZE, 1> Gradient_imp() const {
    return OperatorImp::Gradient_binary_op(lhs_oprand_, rhs_oprand_);
  }

 private:
  const LHS_OPRAND& lhs_oprand_;
  const RHS_OPRAND& rhs_oprand_;
};

class PlusOp {
 public:
 template<typename LHS_OPRAND, typename RHS_OPRAND>
   static auto value_binary_op(const LHS_OPRAND& lhs,
                            const RHS_OPRAND& rhs) {
    return lhs.value() + rhs.value();
  }

 template<typename LHS_OPRAND, typename RHS_OPRAND>
  static auto  Gradient_binary_op(
      const LHS_OPRAND& lhs, const RHS_OPRAND& rhs) {
    return (lhs.Gradient() + rhs.Gradient()).eval();
  }
};

// SFINAE
template <class LHS, class RHS, class U = typename LHS::DATA_TYPE,
          class V = typename RHS::DATA_TYPE>
auto operator+(const LHS& lhs, const RHS& rhs) {
  return BinaryOp<LHS, RHS, PlusOp>(lhs, rhs);
}

class MinusBinaryOp {
 public:
template <class LHS_OPRAND, class RHS_OPRAND>
  static auto value_binary_op(const LHS_OPRAND& lhs,
                            const RHS_OPRAND& rhs) {
    return lhs.value() - rhs.value();
  }
template <class LHS_OPRAND, class RHS_OPRAND>
static auto  Gradient_binary_op(
      const LHS_OPRAND& lhs, const RHS_OPRAND& rhs) {
    return (lhs.Gradient() - rhs.Gradient()).eval();
  }
};
template <class LHS, class RHS, class U = typename LHS::DATA_TYPE,
          class V = typename RHS::DATA_TYPE>
auto operator-(const LHS& lhs, const RHS& rhs) {
  return BinaryOp<LHS, RHS, MinusBinaryOp>(lhs, rhs);
}

class MultipleOp {
 public:
template <class LHS_OPRAND, class RHS_OPRAND>
  static auto value_binary_op(const LHS_OPRAND& lhs,
                            const RHS_OPRAND& rhs) {
    return lhs.value() * rhs.value();
  }
template <class LHS_OPRAND, class RHS_OPRAND>
  static auto Gradient_binary_op(
      const LHS_OPRAND& lhs, const RHS_OPRAND& rhs) {
    return (rhs.value() * lhs.Gradient() + lhs.value() * rhs.Gradient()).eval();
  }
};

template <class LHS, class RHS, class U = typename LHS::DATA_TYPE,
          class V = typename RHS::DATA_TYPE>
auto operator*(const LHS& lhs, const RHS& rhs) {
  return BinaryOp<LHS, RHS, MultipleOp>(lhs, rhs);
}

class DivisionOp {
 public:
template <class LHS_OPRAND, class RHS_OPRAND>
  static auto value_binary_op(const LHS_OPRAND& lhs,
                            const RHS_OPRAND& rhs) {
    return lhs.value() / rhs.value();
  }

template <class LHS_OPRAND, class RHS_OPRAND>
  static auto Gradient_binary_op(
      const LHS_OPRAND& lhs, const RHS_OPRAND& rhs) {
    // f / g = f' * g - g' * f / g / g
    auto&& l = lhs.value();
    auto&& r = rhs.value();
    auto denominator = r * r;
    return (r * lhs.Gradient() - l * rhs.Gradient()) / denominator;
  }
};

template <class LHS, class RHS, typename U = typename LHS::DATA_TYPE,
          class V = typename RHS::DATA_TYPE>
auto operator/(const LHS& lhs, const RHS& rhs) {
  return BinaryOp<LHS, RHS, DivisionOp>(lhs, rhs);
}

template <class RHS, 
          class V = typename RHS::DATA_TYPE, int D = RHS::DUAL_NUMBER_SIZE>
auto operator/(const V& lhs, const RHS& rhs) {
  Jet<V, RHS::DUAL_NUMBER_SIZE> const_lhs(lhs);
  return BinaryOp<decltype(const_lhs), RHS, DivisionOp>(const_lhs, rhs);
}

/*
template <class LHS, class RHS>
std::enable_if_t<std::is_same<typename LHS::DATA_TYPE, typename RHS::DATA_TYPE>::value, DivisionOp<LHS, RHS>> operator/(const LHS& lhs, const RHS& rhs) {
  return DivisionOp<LHS, RHS>(lhs, rhs);
}
*/

template <class Functor, int residual_num, int parameter_num>
class AutoDiffFunction {
 public:
  AutoDiffFunction(Functor&& functor)
      : functor_(std::forward<Functor>(functor)) {}

  Eigen::Matrix<double, residual_num, 1> operator()(
      const Eigen::Matrix<double, parameter_num, 1>& x) const {
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
  Eigen::Matrix<double, residual_num, parameter_num> Gradient(
      Eigen::Matrix<double, parameter_num, 1>& x) const {
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
  Eigen::Matrix<double, parameter_num, residual_num> Jacobian(
      Eigen::Matrix<double, parameter_num, 1>& x) const {
    return Gradient(x).transpose();
  }

 private:
  Functor functor_;
};
#endif  // OPTIMIZATION_JET_H
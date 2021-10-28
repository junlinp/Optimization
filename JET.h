#ifndef OPTIMIZATION_JET_H
#define OPTIMIZATION_JET_H
#include <Eigen/Dense>
#include <type_traits>
#include <assert.h>

// CRTP 
template<class Derive, class Data_Type, int Dimension>
class Expression {
  public:
  using DATA_TYPE = Data_Type;
  static const int DUAL_NUM_SIZE = Dimension;

  DATA_TYPE value() const { return static_cast<const Derive*>(this)->value_imp();}
  Eigen::Matrix<double, DUAL_NUM_SIZE, 1> Gradient() const { return static_cast<const Derive*>(this)->Gradient_imp();}

  friend Derive;
};

// BASIC_TYPE should be float or double
// and there is a problem to initialize a array of Jet.
template <class BASIC_TYPE, int N>
class Jet : public Expression<Jet<BASIC_TYPE, N>, BASIC_TYPE, N> {
 public:
  using DATA_TYPE = BASIC_TYPE;
  static const int DUAL_NUMBER_SIZE = N;
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
  template<class EXPR>
  Jet& operator=(const EXPR& expr) {
    value_ = expr.value();
    gradient_ = expr.Gradient();
    return *this;
  }

  template<class EXPR>
  Jet(const EXPR& expr) : value_(expr.value()), gradient_(expr.Gradient()) {}

  void setGradient(size_t index, BASIC_TYPE value) {
    gradient_(index) = value;
  }
  DATA_TYPE value_imp() const { return value_; }

  Eigen::Matrix<double, N, 1> Gradient_imp() const { return gradient_; }

 private:
  BASIC_TYPE value_;
  Eigen::Matrix<BASIC_TYPE, N, 1> gradient_;
};

template<int N>
using JETF = Jet<float, N>;

template<int N>
using JETD = Jet<double, N>;

template<class LHS_OPRAND, class RHS_OPRAND>
struct Binary_trait {
  static_assert(std::is_same<typename LHS_OPRAND::DATA_TYPE, typename RHS_OPRAND::DATA_TYPE>::value, "The Data Type between LHS Oprand and RHS Oprand is different.");
  static_assert(LHS_OPRAND::DUAL_NUMBER_SIZE == RHS_OPRAND::DUAL_NUMBER_SIZE, "The Dual Number Size between LHS Oprand and RHS Oprand is different.");

  using DATA_TYPE = typename LHS_OPRAND::DATA_TYPE;
  const static int DUAL_NUMBER_SIZE = LHS_OPRAND::DUAL_NUMBER_SIZE;
};

template<class LHS_OPRAND, class RHS_OPRAND, class Derive>
class BinaryOp : public Expression<BinaryOp<LHS_OPRAND, RHS_OPRAND, Derive>, typename Binary_trait<LHS_OPRAND, RHS_OPRAND>::DATA_TYPE, Binary_trait<LHS_OPRAND, RHS_OPRAND>::DUAL_NUMBER_SIZE>{ 
public:
  using DATA_TYPE = typename Binary_trait<LHS_OPRAND, RHS_OPRAND>::DATA_TYPE; 
  static const int DUAL_NUM_SIZE = Binary_trait<LHS_OPRAND, RHS_OPRAND>::DUAL_NUMBER_SIZE;
  BinaryOp(const LHS_OPRAND& lhs, const RHS_OPRAND& rhs) : lhs_oprand_(lhs), rhs_oprand_(rhs){}
  DATA_TYPE value_imp() const { return static_cast<const Derive*>(this)->value_binary_op(lhs_oprand_, rhs_oprand_);}
  Eigen::Matrix<double, DUAL_NUM_SIZE, 1> Gradient_imp() const { return static_cast<const Derive*>(this)->Gradient_binary_op(lhs_oprand_, rhs_oprand_);}
  private:
  const LHS_OPRAND& lhs_oprand_;
  const RHS_OPRAND& rhs_oprand_;
};


template <class LHS_OPRAND, class RHS_OPRAND>
class PlusOp : public BinaryOp<LHS_OPRAND, RHS_OPRAND, PlusOp<LHS_OPRAND, RHS_OPRAND>> {
public:
 using DATA_TYPE = typename LHS_OPRAND::DATA_TYPE;
 static const int DUAL_NUM_SIZE = LHS_OPRAND::DUAL_NUMBER_SIZE; 
 using BASE_CLASS = BinaryOp<LHS_OPRAND, RHS_OPRAND, PlusOp<LHS_OPRAND, RHS_OPRAND>>;

 PlusOp(const LHS_OPRAND& lhs_oprand, const RHS_OPRAND& rhs_oprand) : BinaryOp<LHS_OPRAND, RHS_OPRAND, PlusOp<LHS_OPRAND, RHS_OPRAND>>(lhs_oprand, rhs_oprand) {}
 
  DATA_TYPE value_binary_op(const LHS_OPRAND& lhs, const RHS_OPRAND& rhs) const { return lhs.value() + rhs.value();}
  Eigen::Matrix<double,DUAL_NUM_SIZE, 1> Gradient_binary_op(const LHS_OPRAND& lhs, const RHS_OPRAND& rhs) const { return lhs.Gradient() + rhs.Gradient(); } 
};

template<class LHS, class RHS>
PlusOp<LHS, RHS> operator+(const LHS& lhs, const RHS& rhs) {
  return PlusOp<LHS, RHS>(lhs, rhs);
}

template<class LHS_OPRAND, class RHS_OPRAND>
class MultipleOp {

};
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
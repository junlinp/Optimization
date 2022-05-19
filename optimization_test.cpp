#include "BFGS.h"
#include "JET.h"
#include "LBFGS.h"
#include "gradient_checker.h"
#include "gradient_decent.h"
#include "reduce.h"
#include "gtest/gtest.h"

class Reduce_Add : public ::testing::Test  {
public:
  void SetUp() {
  N = 1024 * 1024 * 32;
  input_arr = new int[N];
  sum = 0;

  for(int i = 0; i < N;i++) {
    input_arr[i] = rand() % 3;
    sum += input_arr[i];
  }

  }

  int N;
  int* input_arr;
  int sum;
};
TEST_F(Reduce_Add, no_avx) {
  // 8 M
  int target_sum = no_avx_reduce(input_arr, N);
  EXPECT_EQ(target_sum, sum);
}

TEST_F(Reduce_Add, avx) {
  // 8 M
  int target_sum = avx_reduce(input_arr, N);
  EXPECT_EQ(target_sum, sum);
}



TEST(JET, Plus) {
  JETD<2> a(1.0, 0);
  JETD<2> b(2.0, 1);

  JETD<2> c = a + b;
  auto gradient = c.Gradient();
  EXPECT_EQ(gradient(0), 1.0);
  EXPECT_EQ(gradient(1), 1.0);
  JETD<2> d = a * a + b * b;
  EXPECT_EQ(d.value(), 5.0);
  EXPECT_EQ(d.Gradient()(0), 2.0);
  EXPECT_EQ(d.Gradient()(1), 4.0);
}

TEST(JET, Sub) {
  JETD<2> a(1.0, 0);
  JETD<2> b(2.0, 1);

  JETD<2> c = a - b;
  auto gradient = c.Gradient();
  EXPECT_EQ(gradient(0), 1.0);
  EXPECT_EQ(gradient(1), -1.0);
}

TEST(JET, Multiple) {
  JETD<2> a(1.0, 0);
  JETD<2> b(2.0, 1);

  JETD<2> c = a * b;
  auto gradient = c.Gradient();
  EXPECT_EQ(gradient(0), 2.0);
  EXPECT_EQ(gradient(1), 1.0);
}

TEST(JET, Eigen_Map) {
  std::vector<JETD<2>> input(2);
  for (int i = 0; i < 2; i++) {
    input[i] = JETD<2>(i + 1, i);
  }
  auto eigen = Eigen::Map<Eigen::Matrix<JETD<2>, 2, 1>>(&input[0]);
  auto first_element = eigen(0);
  auto second_element = eigen(1);
  EXPECT_EQ(first_element.value(), 1.0);
  EXPECT_EQ(second_element.value(), 2.0);
  EXPECT_EQ(first_element.Gradient()(0), 1.0);
  EXPECT_EQ(first_element.Gradient()(1), 0.0);
  EXPECT_EQ(second_element.Gradient()(0), 0.0);
  EXPECT_EQ(second_element.Gradient()(1), 1.0);

  Eigen::Matrix<JETD<2>, 1, 1> result(eigen(0) * eigen(0) + eigen(1) * eigen(1));

  EXPECT_NEAR(result(0).value(), 5.0, 1e-5);
  auto g = result(0).Gradient();
  EXPECT_NEAR(g(0), 2.0, 1e-5);
  EXPECT_NEAR(g(1), 4.0, 1e-5);
}

TEST(JET, Eigen_Multiple) {
  // the initializer convert to JETD is invalid operator
  // should support the implicit convert from 
  Eigen::Matrix<JETD<2>, 2, 2> A;
  A << 1.0, 2.0, 
       3.0, 4.0;
  Eigen::Matrix<JETD<2>, 2, 2> b, x;
  x << 1.0, 0.0;
  // BUG : Eigen multiple operator overload error
  // b = 1.0 * A * x;

  EXPECT_NEAR(b(0).value(), 1.0, 1e-5);
  EXPECT_NEAR(b(1).value(), 3.0, 1e-5);
}

struct LinearFunctor {
  template <class T>
  bool operator()(T* input, T* residual) const {
    residual[0] = T(2.0) * input[0] + T(3.0) * input[1];
    return true;
  }
};

TEST(JET, FUNCTOR_Wrap) {
  std::vector<double> x = {1.0, 2.0};
  std::vector<double> res = {0.0};
  LinearFunctor origin_functor;
  origin_functor(x.data(), res.data());
  EXPECT_EQ(res[0], 1.0 * 2.0 + 2.0 * 3.0);
  AutoDiffFunction<LinearFunctor, 1, 2> auto_diff_functor(
      std::move(origin_functor));
  Eigen::Matrix<double, 2, 1> eigen_x;
  eigen_x << 1.0, 2.0;
  Eigen::Matrix<double, 1, 1> eigen_res(0.0);
  eigen_res = auto_diff_functor(eigen_x);
  EXPECT_EQ(eigen_res(0), 1.0 * 2.0 + 2.0 * 3.0);
}

struct LinearSystem {
  LinearSystem() {}
  template <class T>
  bool operator()(T* input, T* residual) const {
    Eigen::Matrix<T, 2, 1> x = Eigen::Map<Eigen::Matrix<T, 2, 1>>(input);

    Eigen::Matrix<T, 2, 2> A;
    Eigen::Matrix<T, 2, 1> b;
    A << 5.0, 7.0, 7.0, 11.0;
    b << 31.0, 47.0;
    
    //Eigen::Matrix<T, 2, 1>  error = (A * x - b);
    T error[2];
    error[0] = A(0, 0) * x(0) + A(0, 1) * x(1) - b(0);
    error[1] = A(1, 0) * x(0) + A(1, 1) * x(1) - b(1);
    
    residual[0] = error[0] * error[0] + error[1] * error[1];
    return true;
  }
};
TEST(LenearSystem, Gradient_CHECK) {
  LinearSystem functor;
  AutoDiffFunction<LinearSystem, 1, 2> auto_diff(std::move(functor));
  Eigen::Matrix<double, 2, 1> x;
  x << 3.4181, 2.06285;
  for (int i = 0; i < 4; i++) {
    EXPECT_TRUE(GradientChecker(auto_diff, x));
    //auto gradient = auto_diff.Gradient(x);

    auto computed_gradient = [](auto x0) {
      Eigen::Matrix<double, 2, 2> ATA;
      ATA << 74, 112, 112, 170;
      Eigen::Matrix<double, 2, 1> ATb;
      ATb << 484, 734;
      return 2 * ATA * x0 - 2 * ATb;
    };
	/*
    auto inference = [](auto x0) {
      Eigen::Matrix<double, 2, 2> A;
      A << 5, 7, 7, 11;
      Eigen::Matrix<double, 2, 1> b;
      b << 31, 47;
      return (A * x0 - b).dot(A * x0 - b);
    };
    */
    auto computed_gradient_value = computed_gradient(x);
    // std::cout << "gradient : " << gradient << std::endl;
    // std::cout << "computed_gradient : " << computed_gradient_value <<
    // std::endl; std::cout << "V : " << auto_diff(x) << std::endl; std::cout <<
    // "Value : " << inference(x) << std::endl;
    x = x - 0.001 * computed_gradient_value;
    // std::cout << "x : " << x.transpose() << std::endl;
  }
}
TEST(LenearSystem, Gradient_Decent) {
  LinearSystem functor;
  AutoDiffFunction<LinearSystem, 1, 2> auto_diff(std::move(functor));
  Eigen::Matrix<double, 2, 1> x;
  x << 0.0, 0.0;
  GradientDecent<LinearSystem, 1, 2>(auto_diff, x);
  EXPECT_NEAR(x(0), 2.0, 1e-9);
  EXPECT_NEAR(x(1), 3.0, 1e-9);
}

TEST(LinearSystem, Newton_Method_BFGS) {
  LinearSystem functor;
  AutoDiffFunction<LinearSystem, 1, 2> auto_diff(std::move(functor));
  Eigen::Matrix<double, 2, 1> x;
  x << 0.0, 0.0;
  BFGS(auto_diff, x);
  EXPECT_NEAR(x(0), 2.0, 1e-9);
  EXPECT_NEAR(x(1), 3.0, 1e-9);
}

TEST(LinearSystem, Newton_Method_LBFGS) {
  LinearSystem functor;
  AutoDiffFunction<LinearSystem, 1, 2> auto_diff(std::move(functor));
  Eigen::Matrix<double, 2, 1> x;
  x << 0.0, 0.0;
  LBFGS(auto_diff, x);
  EXPECT_NEAR(x(0), 2.0, 1e-9);
  EXPECT_NEAR(x(1), 3.0, 1e-9);
}

int main() {
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}

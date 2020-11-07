#include "gtest/gtest.h"
#include "evaluate.h"

TEST(Project_Function, simple_case) {
  double camera[9] = {
      0.0157415, -0.0127909, -0.00440085,  -0.0340938,  -0.107514,
      1.12022,   399.752,    -3.17706e-07, 5.88205e-13,
  };
  double point[3] = {-0.612, 0.571759, -1.84708};
  double uv[2] = {-332.65, 262.09};

  ProjectFunction p(uv[0], uv[1]);
  double residual[2];
  p(camera, point, residual);
  std::cout << "Residual[0] : " << residual[0] << std::endl;
  std::cout << "Residual[1] : " << residual[1] << std::endl;
}

TEST(Project_Function, Jet) {
    Jet<12> param[12];
    double p[12] = {0.0157415,   -0.0127909, -0.00440085, -0.0340938,
                    -0.107514,   1.12022,    399.752,     -3.17706e-07,
                    5.88205e-13, -0.612,     0.571759,    -1.84708};
    for(int i = 0; i < 12; i++) {
        param[i] = Jet<12>(p[i], i);
    }
    Jet<12> residual[2];
    ProjectFunction functor(-332.65, 262.09);

    functor(param, param + 9, residual);

    std::cout << "Residual[0] : " << residual[0].value() << std::endl;
    std::cout << "Residual[1] : " << residual[1].value() << std::endl;
}

int main() {
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
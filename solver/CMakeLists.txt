cmake_minimum_required(VERSION 3.13)

set(CXX_STANDARD 17)
find_package(Eigen3 REQUIRED)

add_library(solver lm.cc)
target_link_libraries(solver Eigen3::Eigen)

find_package(GTest REQUIRED)
#add_executable(solver_unittest solver_test.cc)
#target_link_libraries(solver_unittest GTest::GTest Eigen3::Eigen solver)
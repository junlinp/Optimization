cmake_minimum_required(VERSION 3.0)

project(bal)
set(CXX_STANDARD 11)
add_library(${PROJECT_NAME}_lib load_problem.cc evaluate.cc)
target_link_libraries(${PROJECT_NAME}_lib Eigen3::Eigen)

add_executable(${PROJECT_NAME} bal_main.cc )
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen ceres ${PROJECT_NAME}_lib)

find_package(GTest REQUIRED)
add_executable(bal_unittest bal_test.cc evaluate.cc)
target_link_libraries(bal_unittest GTest::GTest ${PROJECT_NAME}_lib Eigen3::Eigen)
cmake_minimum_required(VERSION 3.0)

project(bal)
set(CXX_STANDARD 20)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

add_library(${PROJECT_NAME}_lib load_problem.cc ceres_bal_solver.cc daba_bal_solver.cc daba_subproblem.cc daba_subproblem_manager.cc)
target_link_libraries(${PROJECT_NAME}_lib Eigen3::Eigen ceres Graph)

add_executable(${PROJECT_NAME} bal_main.cc )

target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen ceres ${PROJECT_NAME}_lib)

find_package(GTest REQUIRED)
add_executable(bal_unittest bal_test.cc)
target_link_libraries(bal_unittest GTest::GTest Eigen3::Eigen ceres)
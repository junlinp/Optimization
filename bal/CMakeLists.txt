cmake_minimum_required(VERSION 3.0)

project(bal)
set(CXX_STANDARD 11)

add_executable(${PROJECT_NAME} bal_main.cc load_problem.cc evaluate.cc)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen ceres)
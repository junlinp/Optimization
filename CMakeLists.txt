cmake_minimum_required(VERSION 3.13)
project(Optimization)

find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
set(CMAKE_CXX_STANDARD 11)
add_executable(${PROJECT_NAME}_test optimization_test.cpp)

target_link_libraries(${PROJECT_NAME}_test PUBLIC
GTest::GTest
Eigen3::Eigen
)
cmake_minimum_required(VERSION 3.0)

project(first_order_methods)
set(CXX_STANDARD 17)
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME}_lib gradient_descent.cc)
target_link_libraries(${PROJECT_NAME}_lib Eigen3::Eigen)

find_package(GTest REQUIRED)

add_executable(${PROJECT_NAME}_unittest ${PROJECT_NAME}_test.cc)
target_link_libraries(${PROJECT_NAME}_unittest GTest::GTest Eigen3::Eigen ${PROJECT_NAME}_lib)
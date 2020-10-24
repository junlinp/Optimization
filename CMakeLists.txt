cmake_minimum_required(VERSION 3.13)
project(Optimization)

find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(GLOG REQUIRED)
set(CMAKE_CXX_STANDARD 14)
add_library(mmio mmio.c mmio.h)
add_executable(${PROJECT_NAME}_test optimization_test.cpp)

target_link_libraries(${PROJECT_NAME}_test PUBLIC
GTest::GTest
Eigen3::Eigen
glog::glog
)

add_executable(Linear_Solver_test linear_solver_test.cpp)
target_link_libraries(Linear_Solver_test PUBLIC
GTest::GTest
Eigen3::Eigen
glog::glog
)

add_compile_options(-Wall -Wextra -pedantic -Werror)

add_executable(Sparse_Matrix_Test sparse_matrix_test.cpp )
target_link_libraries(Sparse_Matrix_Test GTest::GTest mmio)
target_compile_definitions(Sparse_Matrix_Test
    PRIVATE -DTest_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

add_executable(IterativeMethodTest itertive_method_test.cpp)
target_link_libraries(IterativeMethodTest GTest::GTest)

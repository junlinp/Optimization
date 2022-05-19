cmake_minimum_required(VERSION 3.13)
project(Optimization)

find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(GLOG REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-mavx -mavx2 -g -O3)
add_library(mmio mmio.c mmio.h)

add_executable(${PROJECT_NAME}_test optimization_test.cpp reduce.cc)
target_link_libraries(${PROJECT_NAME}_test PUBLIC
GTest::GTest
Eigen3::Eigen
glog::glog
)
set(HAMMING_DATASET_PATH "${CMAKE_SOURCE_DIR}/data/Hamming")
configure_file(LinearProgramingConfig.h.in LinearProgramingConfig.h)

add_executable(Linear_Solver_test linear_solver_test.cpp linear_programing.cc)
target_include_directories(Linear_Solver_test PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )

target_link_libraries(Linear_Solver_test PUBLIC
GTest::GTest
Eigen3::Eigen
glog::glog
LAPACK::LAPACK
BLAS::BLAS
)

#add_executable(Auto_Diff_Test auto_diff_test.cc)
#target_link_libraries(Auto_Diff_Test PUBLIC
#GTest::GTest
#Eigen3::Eigen
#)

add_compile_options(-Wall -Wextra -pedantic -Werror)

add_executable(Sparse_Matrix_Test sparse_matrix_test.cpp )
target_link_libraries(Sparse_Matrix_Test GTest::GTest mmio)
target_compile_definitions(Sparse_Matrix_Test
    PRIVATE -DTest_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

add_executable(IterativeMethodTest itertive_method_test.cpp)
target_link_libraries(IterativeMethodTest GTest::GTest)


#add_subdirectory(bal)
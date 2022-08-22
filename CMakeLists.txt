cmake_minimum_required(VERSION 3.13)
project(Optimization)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

#find_package(GLOG REQUIRED)
#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-mavx -mavx2 -g -O3)
add_compile_options(-Wall -Wextra -pedantic )
add_library(mmio IO/mmio.c IO/mmio.h)

BISON_TARGET(mps_parser IO/mps.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp)
flex_target(mps_scanner IO/mps.lex ${CMAKE_CURRENT_BINARY_DIR}/lexer.cpp)
ADD_FLEX_BISON_DEPENDENCY(mps_scanner mps_parser)
add_library(mps_io IO/mps_format_io.h IO/mps_format_io.cc ${BISON_mps_parser_OUTPUTS} ${FLEX_mps_scanner_OUTPUTS}) 
target_include_directories(mps_io PUBLIC IO ${CMAKE_CURRENT_BINARY_DIR})

add_library(LinearPrograming linear_programing.cc)
target_link_libraries(LinearPrograming Eigen3::Eigen)

add_executable(LPSolver lp_solver.cpp)
target_link_libraries(LPSolver PUBLIC mps_io ${FLEX_LIBRARIES} LinearPrograming)

#add_executable(${PROJECT_NAME}_test optimization_test.cpp reduce.cc)
#target_link_libraries(${PROJECT_NAME}_test PUBLIC
#GTest::GTest
#Eigen3::Eigen
#glog::glog
#)

set(HAMMING_DATASET_PATH "${CMAKE_SOURCE_DIR}/data/Hamming")
configure_file(LinearProgramingConfig.h.in LinearProgramingConfig.h)

add_executable(Linear_Solver_test linear_solver_test.cpp)
target_include_directories(Linear_Solver_test PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )

target_link_libraries(Linear_Solver_test PUBLIC
GTest::GTest
Eigen3::Eigen
LinearPrograming
#LAPACK::LAPACK
#BLAS::BLAS
)

#add_executable(Auto_Diff_Test auto_diff_test.cc)
#target_link_libraries(Auto_Diff_Test PUBLIC
#GTest::GTest
#Eigen3::Eigen
#)


#add_executable(Sparse_Matrix_Test sparse_matrix_test.cpp )
#target_link_libraries(Sparse_Matrix_Test GTest::GTest mmio
#Eigen3::Eigen)
#target_compile_definitions(Sparse_Matrix_Test
#    PRIVATE -DTest_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

#add_executable(IterativeMethodTest itertive_method_test.cpp)
#target_link_libraries(IterativeMethodTest GTest::GTest
#Eigen3::Eigen)


#add_subdirectory(bal)

cmake_minimum_required(VERSION 3.13)
project(Optimization)
set (CMAKE_EXPORT_COMPILE_COMMANDS True) 
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-mavx -mavx2 -g -O3)
add_compile_options(-Wall -Wextra -pedantic )


#add_executable(LPSolver lp_solver.cpp)
#target_link_libraries(LPSolver PUBLIC mps_io ${FLEX_LIBRARIES} LinearPrograming)

#add_executable(${PROJECT_NAME}_test optimization_test.cpp reduce.cc)
#target_link_libraries(${PROJECT_NAME}_test PUBLIC
#GTest::GTest
#Eigen3::Eigen
#glog::glog
#)


#add_executable(Linear_Solver_test linear_solver_test.cpp)

#target_include_directories(Linear_Solver_test PUBLIC
#                           "${PROJECT_BINARY_DIR}"
#                           )

#target_link_libraries(Linear_Solver_test PUBLIC
#GTest::GTest
#Eigen3::Eigen
#)

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


add_subdirectory(bal)
add_subdirectory(Graph)
add_subdirectory(RGD)
add_subdirectory(solver)
add_subdirectory(first_order_methods)
add_subdirectory(programing)

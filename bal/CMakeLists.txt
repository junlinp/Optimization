cmake_minimum_required(VERSION 3.0)

project(bal)
set(CXX_STANDARD 20)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

add_library(${PROJECT_NAME}_lib load_problem.cc ceres_bal_solver.cc daba_bal_solver.cc daba_subproblem.cc daba_subproblem_manager.cc)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC Eigen3::Eigen ceres Graph)

# Link BLAS libraries if available
if(APPLE AND TARGET Accelerate::Accelerate)
    target_link_libraries(${PROJECT_NAME}_lib PUBLIC Accelerate::Accelerate)
elseif(UNIX AND NOT APPLE AND TARGET BLAS::BLAS)
    target_link_libraries(${PROJECT_NAME}_lib PUBLIC BLAS::BLAS)
endif()

add_executable(${PROJECT_NAME} bal_main.cc )

target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen ceres ${PROJECT_NAME}_lib rgd_lib)

# Link BLAS libraries if available
if(APPLE AND TARGET Accelerate::Accelerate)
    target_link_libraries(${PROJECT_NAME} PRIVATE Accelerate::Accelerate)
elseif(UNIX AND NOT APPLE AND TARGET BLAS::BLAS)
    target_link_libraries(${PROJECT_NAME} PRIVATE BLAS::BLAS)
endif()

find_package(GTest REQUIRED)
add_executable(bal_unittest bal_test.cc)
target_link_libraries(bal_unittest PUBLIC GTest::GTest Eigen3::Eigen ceres)

# Link BLAS libraries if available
if(APPLE AND TARGET Accelerate::Accelerate)
    target_link_libraries(bal_unittest PRIVATE Accelerate::Accelerate)
elseif(UNIX AND NOT APPLE AND TARGET BLAS::BLAS)
    target_link_libraries(bal_unittest PRIVATE BLAS::BLAS)
endif()
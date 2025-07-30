cmake_minimum_required(VERSION 3.13)

set(CXX_STANDARD 17)
find_package(Eigen3 REQUIRED)

add_library(solver lm.cc)
target_link_libraries(solver PUBLIC Eigen3::Eigen)

# Link BLAS libraries if available
if(APPLE AND TARGET Accelerate::Accelerate)
    target_link_libraries(solver PUBLIC Accelerate::Accelerate)
elseif(UNIX AND NOT APPLE AND TARGET BLAS::BLAS)
    target_link_libraries(solver PUBLIC BLAS::BLAS)
endif()

find_package(GTest REQUIRED)
#add_executable(solver_unittest solver_test.cc)
#target_link_libraries(solver_unittest GTest::GTest Eigen3::Eigen solver)
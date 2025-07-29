cmake_minimum_required(VERSION 3.13)

set(CXX_STANDARD 17)
find_package(Eigen3 REQUIRED)

add_library(solver lm.cc)
target_link_libraries(solver PUBLIC Eigen3::Eigen)

# Link Accelerate framework if available (for Eigen3 BLAS support)
if(APPLE AND TARGET Accelerate::Accelerate)
    target_link_libraries(solver PUBLIC Accelerate::Accelerate)
endif()

find_package(GTest REQUIRED)
#add_executable(solver_unittest solver_test.cc)
#target_link_libraries(solver_unittest GTest::GTest Eigen3::Eigen solver)
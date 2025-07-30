cmake_minimum_required(VERSION 3.0)

project(first_order_methods)
set(CXX_STANDARD 17)
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME}_lib gradient_descent.cc)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC Eigen3::Eigen)

# Link BLAS libraries if available
if(APPLE AND TARGET Accelerate::Accelerate)
    target_link_libraries(${PROJECT_NAME}_lib PUBLIC Accelerate::Accelerate)
elseif(UNIX AND NOT APPLE AND TARGET BLAS::BLAS)
    target_link_libraries(${PROJECT_NAME}_lib PUBLIC BLAS::BLAS)
endif()

find_package(GTest REQUIRED)

add_executable(${PROJECT_NAME}_unittest ${PROJECT_NAME}_test.cc)
target_link_libraries(${PROJECT_NAME}_unittest PUBLIC GTest::GTest Eigen3::Eigen ${PROJECT_NAME}_lib)

# Link BLAS libraries if available
if(APPLE AND TARGET Accelerate::Accelerate)
    target_link_libraries(${PROJECT_NAME}_unittest PRIVATE Accelerate::Accelerate)
elseif(UNIX AND NOT APPLE AND TARGET BLAS::BLAS)
    target_link_libraries(${PROJECT_NAME}_unittest PRIVATE BLAS::BLAS)
endif()
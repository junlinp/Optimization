
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

add_library(linear_programing linear_programing.cc)
target_link_libraries(linear_programing PUBLIC Eigen3::Eigen)

add_executable(linear_programing_test linear_solver_test.cpp)
target_link_libraries(linear_programing_test linear_programing gtest)
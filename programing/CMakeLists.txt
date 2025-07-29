
# Find OpenMP for multi-threading
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP found and enabled")
else()
    message(WARNING "OpenMP not found - Eigen will run single-threaded")
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

add_library(linear_programing linear_programing.cc mps_reader.cc)
target_link_libraries(linear_programing PUBLIC Eigen3::Eigen)

# Link Accelerate framework if available (for Eigen3 BLAS support)
if(APPLE AND TARGET Accelerate::Accelerate)
    target_link_libraries(linear_programing PUBLIC Accelerate::Accelerate)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(linear_programing PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(linear_programing_test linear_solver_test.cpp)
target_link_libraries(linear_programing_test PUBLIC linear_programing gtest)

# Link Accelerate framework if available (for Eigen3 BLAS support)
if(APPLE AND TARGET Accelerate::Accelerate)
    target_link_libraries(linear_programing_test PRIVATE Accelerate::Accelerate)
endif()

add_executable(solve_mps solve_mps.cc)
target_link_libraries(solve_mps PRIVATE linear_programing)

# Link Accelerate framework if available (for Eigen3 BLAS support)
if(APPLE AND TARGET Accelerate::Accelerate)
    target_link_libraries(solve_mps PRIVATE Accelerate::Accelerate)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(solve_mps PRIVATE OpenMP::OpenMP_CXX)
endif()
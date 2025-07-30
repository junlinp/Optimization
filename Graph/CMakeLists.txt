
add_library(Graph graph.cc graph_normal_cut.cc)

find_package(Eigen3 REQUIRED)
target_link_libraries(Graph PUBLIC Eigen3::Eigen)

# Link BLAS libraries if available
if(APPLE AND TARGET Accelerate::Accelerate)
    target_link_libraries(Graph PUBLIC Accelerate::Accelerate)
elseif(UNIX AND NOT APPLE AND TARGET BLAS::BLAS)
    target_link_libraries(Graph PUBLIC BLAS::BLAS)
endif()

target_include_directories(Graph PUBLIC ${CMAKE_SOURCE_DIR})

find_package(GTest REQUIRED)

add_executable(Graph_Unittest graph_test_main.cc)
target_link_libraries(Graph_Unittest PUBLIC GTest::GTest Graph)

# Link BLAS libraries if available
if(APPLE AND TARGET Accelerate::Accelerate)
    target_link_libraries(Graph_Unittest PRIVATE Accelerate::Accelerate)
elseif(UNIX AND NOT APPLE AND TARGET BLAS::BLAS)
    target_link_libraries(Graph_Unittest PRIVATE BLAS::BLAS)
endif()

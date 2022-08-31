
add_library(Graph graph.cc graph_normal_cut.cc)

find_package(Eigen3 REQUIRED)
target_link_libraries(Graph PUBLIC Eigen3::Eigen)

find_package(GTest REQUIRED)

add_executable(Graph_Unittest graph_test_main.cc)
target_link_libraries(Graph_Unittest PUBLIC  GTest::GTest Graph)

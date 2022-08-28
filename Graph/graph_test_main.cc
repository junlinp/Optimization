#include "gtest/gtest.h"

#include "graph_normal_cut.h"

TEST(GraphNormalCut, SimpleCast) {
    Graph graph(7);

    graph.SetEdgeValue(0, 1, 2);
    graph.SetEdgeValue(0, 2, 1);
    graph.SetEdgeValue(0, 3, 1);
    graph.SetEdgeValue(1, 2, 1);
    graph.SetEdgeValue(1, 3, 1);
    graph.SetEdgeValue(2, 3, 2);
    graph.SetEdgeValue(2, 4, 1);
    graph.SetEdgeValue(3, 4, 2);
    graph.SetEdgeValue(3, 5, 1);
    graph.SetEdgeValue(4, 5, 1);
    graph.SetEdgeValue(4, 6, 1);
    graph.SetEdgeValue(5, 6, 2);
    GraphNormalCut cut_solution;

    auto&& [a_set, b_set] = cut_solution.Cut(graph);

    ASSERT_EQ(a_set.size(), 3);
    for(size_t index : a_set) {
        EXPECT_GT(index, 3);
    }
    ASSERT_EQ(b_set.size(), 4);
    for(size_t index : b_set) {
        EXPECT_LE(index, 3);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

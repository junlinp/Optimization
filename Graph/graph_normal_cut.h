#ifndef GRAPH_GRAPH_NORMAL_CUT_H_
#define GRAPH_GRAPH_NORMAL_CUT_H_

#include <set>

#include "graph.h"

class GraphNormalCut {
public:
    using IndexSet = std::set<size_t>;
    using AIndex = IndexSet;
    using BIndex = IndexSet;
    std::pair<AIndex, BIndex> Cut(const Graph& graph) const;
};

#endif //  GRAPH_GRAPH_NORMAL_CUT_H_
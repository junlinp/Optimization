#include "graph.h"

#include "iostream"

Graph::Graph(size_t node_size) : node_size_{node_size}, weight_{}{}

void Graph::SetEdgeValue(size_t lhs_node_id, size_t rhs_node_id, double value) {
    weight_[CreatePair(lhs_node_id, rhs_node_id)] = value;
}

double Graph::GetEdgeValue(size_t lhs_node_id, size_t rhs_node_id) const {
    auto p = CreatePair(lhs_node_id, rhs_node_id);
    if (weight_.find(p) == weight_.end()) {
        return 0.0;
    }
    return weight_.at(p);
}

Graph::NodePair Graph::CreatePair(size_t lhs_node_id, size_t rhs_node_id) const {
    return {
        std::min(lhs_node_id, rhs_node_id),
        std::max(lhs_node_id, rhs_node_id)
    };
}

Eigen::MatrixXd Graph::GetWeightMatrix() const {
    Eigen::MatrixXd res(node_size_, node_size_);
    res.setZero();
    std::cout << "Weight : " << weight_.size() << std::endl;
    // there is performance critial
    for (auto&& [index_piar, weight] : weight_) {
        res(index_piar.first, index_piar.second) = weight;
        res(index_piar.second, index_piar.first) = weight;
    }
    return res;
}
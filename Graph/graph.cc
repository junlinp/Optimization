#include "graph.h"

#include "iostream"

Graph::Graph(size_t node_size) : node_size_{node_size}, weight_{}{}

void Graph::SetEdgeValue(size_t lhs_node_id, size_t rhs_node_id, double value) {
    weight_[CreatePair(lhs_node_id, rhs_node_id)] = value;
}

double Graph::GetEdgeValue(size_t lhs_node_id, size_t rhs_node_id) const {
    return weight_.at(CreatePair(lhs_node_id, rhs_node_id));
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
    for (auto&& [index_piar, weight] : weight_) {
        res(index_piar.first, index_piar.second) = weight;
        res(index_piar.second, index_piar.first) = weight;
    }
    return res;
}
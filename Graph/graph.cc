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

Eigen::SparseMatrix<float> Graph::GetSparseWeightMatrix() const {
    Eigen::SparseMatrix<float> res(node_size_, node_size_);
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(weight_.size());
    for (auto&& [index_piar, weight] : weight_) {
        auto&& i = index_piar.first;
        auto&& j = index_piar.second;
      tripletList.push_back(T(i, j, weight));
      tripletList.push_back(T(j, i, weight));
    }
    res.setFromTriplets(tripletList.begin(), tripletList.end());
    return res;
}
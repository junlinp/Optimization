#ifndef GRAPH_GRAPH_H_
#define GRAPH_GRAPH_H_

#include <map>

#include "Eigen/Dense"
#include "Eigen/Sparse"

class Graph {
public:
   Graph(size_t node_size);

  void SetEdgeValue(size_t lhs_node_id, size_t rhs_node_id, double value);
  double GetEdgeValue(size_t lhs_node_id, size_t rhs_node_id) const;
  Eigen::MatrixXd GetWeightMatrix() const;
  Eigen::SparseMatrix<float> GetSparseWeightMatrix() const;

private:
  size_t node_size_;
  using NodePair = std::pair<size_t, size_t>;
  NodePair CreatePair(size_t lhs_node_id, size_t rhs_node_id) const;
  std::map<NodePair, double> weight_;
};

#endif // GRAPH_GRAPH_H_

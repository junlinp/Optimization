#include "graph_normal_cut.h"
#include <iostream>
std::pair<GraphNormalCut::AIndex, GraphNormalCut::BIndex>
GraphNormalCut::Cut(const Graph &graph) const {

  Eigen::MatrixXd W = graph.GetWeightMatrix();
  size_t n = W.rows();
  Eigen::VectorXd d(n);
  d = W.colwise().sum();

  Eigen::MatrixXd D = d.asDiagonal();
  Eigen::MatrixXd D_sqrt = d.array().sqrt().matrix().asDiagonal();

  Eigen::MatrixXd A = D_sqrt.inverse() * (D - W) * D_sqrt.inverse();
  Eigen::MatrixXd V = A.bdcSvd(Eigen::ComputeFullV).matrixV();

  Eigen::VectorXd solution = V.row(n - 2);

  double mean = solution.mean();

  AIndex a_set;
  BIndex b_set;
  for (size_t i = 0; i < n; i++) {

    if (solution(i) > mean) {
      a_set.insert(i);
    } else {
      b_set.insert(i);
    }
  }
  return {a_set, b_set};
}
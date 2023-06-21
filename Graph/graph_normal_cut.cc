#include "graph_normal_cut.h"

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
      a_set.push_back(i);
    } else {
      b_set.push_back(i);
    }
  }
  return {a_set, b_set};
}

Eigen::VectorXd RiemannProject(const Eigen::VectorXd &x,
                               const Eigen::VectorXd &v) {
  return v - v.dot(x) * x;
}

Eigen::VectorXd EuclideanProject(const Eigen::VectorXd & /*x*/,
                                const Eigen::VectorXd &v) {
  return v;
}

Eigen::VectorXd RiemannRetraction(const Eigen::VectorXd& x, const Eigen::VectorXd& step){

  Eigen::VectorXd move = x + step;
  return move / move.norm();
}


Eigen::VectorXd RayleighQuotient(
    const Eigen::SparseMatrix<float> &H,
    Eigen::VectorXd eigenvector_with_smallest_eigenvalue) {
      size_t n = H.rows();
      assert(H.rows() == H.cols());
      
      size_t max_iter = 128;
      size_t iter = 0;
      Eigen::VectorXd x = Eigen::VectorXd::Random(n);
      double lambda = 0.0;
      while(iter < max_iter) {
        Eigen::VectorXd grad_x = H * x + eigenvector_with_smallest_eigenvalue * lambda;
        double grad_lambda = eigenvector_with_smallest_eigenvalue.dot(x);

        // Project gradient to tangent space
        Eigen::VectorXd riemann_grad_x = RiemannProject(x, grad_x);
        // Euclidean Project is identity map
        double riemann_grad_lambda = grad_lambda;

        // stopping criterion

        // linear search for step-size
        double step = 1.0;


        x = RiemannRetraction(x, -step * riemann_grad_x);
        lambda = lambda - grad_lambda * step;
      }
    }

std::pair<GraphNormalCut::AIndex, GraphNormalCut::BIndex>
GraphNormalCut::SparseCut(const Graph& graph) const {
  Eigen::SparseMatrix<float> W = graph.GetSparseWeightMatrix();
  size_t n = W.rows();
  Eigen::VectorXd d(n);
  for (int k = 0; k < W.outerSize(); ++k) {
    for (Eigen::SparseMatrix<float>::InnerIterator it(W, k); it; ++it) {
      it.value();
      it.row();   // row index
      it.col();   // col index (here it is equal to k)
      it.index(); // inner index, here it is equal to it.row()
      d(it.row()) += it.value();
    }
  }
  Eigen::VectorXd d_sqrt_invert = d.array().sqrt().inverse().matrix();
  Eigen::SparseMatrix<float> D_minus_W = -W;
  for (int i= 0; i < n; i++) {
    D_minus_W.coeffRef(i, i) += d(i);
  }
  // A = d_sqrt_invert.asDiagonal() * D_minus_W * d_sqrt_invert.asdiagonal();
  Eigen::SparseMatrix<float> A = D_minus_W;

  // A = RightMultiple(A, d_sqrt_invert);
  // A = LeftMultiple(A, d_sqrt_invert);

  for (int k = 0; k < A.outerSize(); k++) {
    for (Eigen::SparseMatrix<float>::InnerIterator it(A, k); it; ++it) {
      it.valueRef() *= d_sqrt_invert(it.row()) * d_sqrt_invert(it.col());
    }
  }


  Eigen::VectorXd eigen_vector_with_small_eigenvalue;
  // Fiedler vector
  // Rayleigh quotient
  Eigen::VectorXd solution = RayleighQuotient(A, eigen_vector_with_small_eigenvalue);

  double mean = solution.mean();

  AIndex a_set;
  BIndex b_set;
  for (size_t i = 0; i < n; i++) {

    if (solution(i) > mean) {
      a_set.push_back(i);
    } else {
      b_set.push_back(i);
    }
  }
  return {a_set, b_set};
}
#include "graph_normal_cut.h"

#include <iostream>
#include <limits>

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

template<class VectorT>
VectorT RiemannProject(const VectorT &x,
                               const VectorT &v,
                               const VectorT &s
                               ) {
  VectorT t =  v - v.dot(x) * x;
  return t - t.dot(s) * s;
}

template<class VectorT>
VectorT RiemannRetraction(const VectorT& x, const VectorT& step){
  VectorT move = x + step;
  return move / move.norm();
}

template<class MatrixT, class VectorT>
float BackTrackingSearch(const MatrixT& H,
                          const VectorT& x, const VectorT& grad,
                          double initial_step) {
  float alpha = initial_step;
  float gramma = 1e-4;
  float tau = 0.5;
  auto F = [&H](const VectorT& x) -> double {
    return 0.5 * x.dot(H * x);
  };
  int iter = 0;
  int max_iter = 1024;
  while(F(x) - F(RiemannRetraction(x, VectorT(-alpha * grad))) <  gramma* alpha * grad.squaredNorm() && iter++ < max_iter) {
    alpha *= tau;
  }
  if (iter == max_iter) {
    std::cout << "BackTrackingSearch reach the max iter" << std::endl;
  }
  return alpha;
}

Eigen::VectorXf RayleighQuotient(
    const Eigen::SparseMatrix<float> &H,
    Eigen::VectorXf eigenvector_with_smallest_eigenvalue) {
      size_t n = H.rows();
      assert(H.rows() == H.cols());
      size_t max_iter = 128;
      size_t iter = 0;
      Eigen::VectorXf x = Eigen::VectorXf::Random(n);
      x = x - x.dot(eigenvector_with_smallest_eigenvalue) * eigenvector_with_smallest_eigenvalue;
      std::cout << "Initial H * x : " << (H*x).norm() << std::endl;
      // minimum f(x)  = 0.5 * x' * H * x
      while(iter++ < max_iter) {
        Eigen::VectorXf grad_x = H * x;

        // Project gradient to tangent space
        Eigen::VectorXf riemann_grad_x = RiemannProject(x, grad_x, eigenvector_with_smallest_eigenvalue);

        // stopping criterion
        if (riemann_grad_x.norm() < 1e-5 * n) {
          break;
        }
        std::cout << "search step" << std::endl;
        // linear search for step-size
        float step = BackTrackingSearch(H, x, riemann_grad_x, 1.0);
        x = RiemannRetraction(x, Eigen::VectorXf(-step * riemann_grad_x));
        std::cout << iter << " iteration" << std::endl;
      }
      std::cout << "H * x : " << (H*x).norm() << std::endl;
      return x;
    }

std::pair<GraphNormalCut::AIndex, GraphNormalCut::BIndex>
GraphNormalCut::SparseCut(const Graph& graph) const {
  Eigen::SparseMatrix<float> W = graph.GetSparseWeightMatrix();
  std::cout << "Construction W finish" << std::endl;
  size_t n = W.rows();
  Eigen::VectorXf d(n);
  d.setZero();
  for (int k = 0; k < W.outerSize(); ++k) {
    for (Eigen::SparseMatrix<float>::InnerIterator it(W, k); it; ++it) {
      it.value();
      it.row();   // row index
      it.col();   // col index (here it is equal to k)
      it.index(); // inner index, here it is equal to it.row()
      d(it.row()) += it.value();
    }
  }
  std::cout << "Construction d" << std::endl;
  Eigen::VectorXf d_sqrt_invert(n);
  Eigen::VectorXf d_sqrt(n);
  for(size_t i = 0; i < n; i++) {
    assert(d(i) > 0);
    d_sqrt_invert(i) = 1.0 / std::sqrt(d(i));
    d_sqrt(i) = std::sqrt(d(i));
  }
   //= d.array().sqrt().inverse();

  std::cout << "Construction finish" << std::endl;
  Eigen::SparseMatrix<float> D_minus_W = -W;
  for (size_t i= 0; i < n; i++) {
    D_minus_W.insert(i, i) = d(i);
  }
  std::cout << "D_minus_W" << std::endl;
  // A = d_sqrt_invert.asDiagonal() * D_minus_W * d_sqrt_invert.asdiagonal();
  Eigen::SparseMatrix<float> A = D_minus_W;

  // A = RightMultiple(A, d_sqrt_invert);
  // A = LeftMultiple(A, d_sqrt_invert);

  for (int k = 0; k < A.outerSize(); k++) {
    for (Eigen::SparseMatrix<float>::InnerIterator it(A, k); it; ++it) {
      float v = it.value();
      assert(!isnan(v));
      v = v * d_sqrt_invert(it.row()) * d_sqrt_invert(it.col());
      assert(!isnan(v));
      it.valueRef() = v;
    }
  }
  std::cout << "A" << std::endl;


  Eigen::VectorXf eigen_vector_with_small_eigenvalue = d_sqrt;
  // Fiedler vector
  // Rayleigh quotient
  Eigen::VectorXf solution = RayleighQuotient(A, eigen_vector_with_small_eigenvalue);
  std::cout << "solution" << std::endl;
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
  std::cout << a_set.size() << " : " << b_set.size() << std::endl;
  return {a_set, b_set};
}
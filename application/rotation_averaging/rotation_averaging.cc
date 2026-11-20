#include "application/rotation_averaging/rotation_averaging.h"

#include "application/rotation_averaging/shonan_certificate.h"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <cmath>
#include <limits>

namespace {

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Quaterniond;
using Eigen::VectorXd;
using std::size_t;

struct Neighbor {
  size_t node;
  Matrix3d to_here;  // Maps a frame at node into the current vertex's frame.
};
using Graph = std::vector<std::vector<Neighbor>>;

// X stacks 3-by-p row-orthonormal frames. Each undirected adjacency pair
// represents one measurement, including when both directions were supplied.
double squaredError(const Graph& graph, const MatrixXd& x) {
  double cost = 0.0;
  for (size_t i = 0; i < graph.size(); ++i) {
    for (const auto& edge : graph[i]) {
      if (i < edge.node) {
        cost += (x.middleRows<3>(3 * i) -
                 edge.to_here * x.middleRows<3>(3 * edge.node)).squaredNorm();
      }
    }
  }
  return cost;
}

MatrixXd neighborSum(const std::vector<Neighbor>& neighbors, const MatrixXd& x) {
  MatrixXd sum = MatrixXd::Zero(3, x.cols());
  for (const auto& edge : neighbors) {
    sum.noalias() += edge.to_here * x.middleRows<3>(3 * edge.node);
  }
  return sum;
}

// At p=3 use the closest proper rotation. For p>3 every row frame can
// be completed to a member of SO(p), so the ordinary polar factor suffices.
MatrixXd projectFrame(const MatrixXd& matrix) {
  Eigen::JacobiSVD<MatrixXd> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Matrix3d u = svd.matrixU();
  if (matrix.cols() == 3 && u.determinant() * svd.matrixV().determinant() < 0) {
    u.col(2) *= -1.0;
  }
  return u * svd.matrixV().transpose();
}

// Right-invariant rotations share a gauge. Holding node zero fixed makes the
// scalar graph Laplacian SPD. It is the chordal Hessian (up to a factor of
// four) at an exact solution and preconditions each of the three coordinates.
int accelerateRotations(const Graph& graph, int budget, MatrixXd* x) {
  const Eigen::Index free_nodes = graph.size() - 1;
  std::vector<Eigen::Triplet<double>> entries;
  for (size_t i = 1; i < graph.size(); ++i) {
    entries.emplace_back(i - 1, i - 1, graph[i].size());
    for (const auto& edge : graph[i]) {
      if (edge.node != 0) entries.emplace_back(i - 1, edge.node - 1, -1.0);
    }
  }
  Eigen::SparseMatrix<double> laplacian(free_nodes, free_nodes);
  laplacian.setFromTriplets(entries.begin(), entries.end());
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> factor(laplacian);
  if (factor.info() != Eigen::Success) return 0;

  double cost = squaredError(graph, *x);
  MatrixXd rhs(free_nodes, 3), trial;
  int attempted = 0;
  while (attempted < budget) {
    ++attempted;
    for (size_t i = 1; i < graph.size(); ++i) {
      const Matrix3d b = x->middleRows<3>(3 * i).transpose() * neighborSum(graph[i], *x);
      rhs.row(i - 1) << (b(2, 1) - b(1, 2)) * 0.5,
                        (b(0, 2) - b(2, 0)) * 0.5,
                        (b(1, 0) - b(0, 1)) * 0.5;
    }
    const MatrixXd direction = factor.solve(rhs);
    const double descent = 4.0 * direction.cwiseProduct(rhs).sum();
    if (!(descent > 0.0) || !std::isfinite(descent)) break;

    bool accepted = false;
    for (double step = 1.0; step >= 1e-6; step *= 0.5) {
      trial = *x;
      for (size_t i = 1; i < graph.size(); ++i) {
        const Eigen::Vector3d omega = direction.row(i - 1).transpose();
        const double angle = omega.norm();
        if (angle > 0.0) {
          trial.middleRows<3>(3 * i) = x->middleRows<3>(3 * i) *
              Eigen::AngleAxisd(step * angle, omega / angle).toRotationMatrix();
        }
      }
      const double next = squaredError(graph, trial);
      if (next < cost && cost - next >= 1e-4 * step * descent) {
        x->swap(trial);
        cost = next;
        accepted = true;
        break;
      }
    }
    if (!accepted) break;
  }
  return attempted;
}

// Exact cyclic block minimization of the lifted chordal objective.
void minimize(const Graph& graph, int max_sweeps, MatrixXd* x) {
  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    double max_change = 0.0;
    for (size_t i = 0; i < graph.size(); ++i) {
      const MatrixXd next = projectFrame(neighborSum(graph[i], *x));
      auto current = x->middleRows<3>(3 * i);
      max_change = std::max(max_change, (next - current).squaredNorm());
      current = next;
    }
    if (max_change < 1e-20) break;
    // Long loops relax slowly under single-node updates. Use part of the
    // local budget for simultaneous, graph-preconditioned SO(3) steps.
    if (sweep == 0 && x->cols() == 3 && graph.size() > 32 && max_sweeps > 2) {
      sweep += accelerateRotations(graph, std::min(24, max_sweeps - 2), x);
    }
  }
}

// For connection Laplacian L, dual blocks D_i = sym((LX)_i X_i^T)
// give sum tr(D_i) = cost(X). If L-D+sI is PSD, cost(X)-3ns is an
// SDP lower bound for every feasible SO(3) solution, even away from a
// stationary point. L's degree blocks cancel in the certificate.
Eigen::SparseMatrix<double> certificateMatrix(const Graph& graph, const MatrixXd& x) {
  std::vector<Eigen::Triplet<double>> entries;
  for (size_t i = 0; i < graph.size(); ++i) {
    const Matrix3d product = neighborSum(graph[i], x) * x.middleRows<3>(3 * i).transpose();
    const Matrix3d diagonal = (product + product.transpose()) * 0.5;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        entries.emplace_back(3 * i + r, 3 * i + c, diagonal(r, c));
      }
    }
    for (const auto& edge : graph[i]) {
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          entries.emplace_back(3 * i + r, 3 * edge.node + c, -edge.to_here(r, c));
        }
      }
    }
  }
  Eigen::SparseMatrix<double> certificate(x.rows(), x.rows());
  certificate.setFromTriplets(entries.begin(), entries.end());
  return certificate;
}

// A negative certificate eigenvector is a descent direction in a fresh
// column. Retract to feasible frames and backtrack until the cost falls.
bool lift(const Graph& graph, const VectorXd& direction, double cost, MatrixXd* x) {
  MatrixXd trial(x->rows(), x->cols() + 1);
  for (double step = 1.0; step >= 1e-8; step *= 0.5) {
    for (size_t i = 0; i < graph.size(); ++i) {
      MatrixXd block(3, trial.cols());
      block.leftCols(x->cols()) = x->middleRows<3>(3 * i);
      block.rightCols<1>() = step * direction.segment<3>(3 * i);
      trial.middleRows<3>(3 * i) = projectFrame(block);
    }
    if (squaredError(graph, trial) < cost) {
      x->swap(trial);
      return true;
    }
  }
  return false;
}

MatrixXd roundToRotations(const Graph& graph, const MatrixXd& x, int max_sweeps) {
  // Keep the three dominant right-singular directions. Their overall
  // handedness is arbitrary, so evaluate both before SO(3) polishing.
  Eigen::SelfAdjointEigenSolver<MatrixXd> eigen(x.transpose() * x);
  MatrixXd projected = x * eigen.eigenvectors().rightCols(3);
  MatrixXd best(x.rows(), 3);
  double best_cost = std::numeric_limits<double>::infinity();
  for (int orientation = 0; orientation < 2; ++orientation) {
    MatrixXd candidate(x.rows(), 3);
    for (Eigen::Index row = 0; row < x.rows(); row += 3) {
      candidate.middleRows<3>(row) = projectFrame(projected.middleRows<3>(row));
    }
    const double cost = squaredError(graph, candidate);
    if (cost < best_cost) {
      best_cost = cost;
      best.swap(candidate);
    }
    projected.col(2) *= -1;
  }
  minimize(graph, max_sweeps, &best);
  return best;
}

// Shonan staircase (Dellaert et al., ECCV 2020, Algorithm 2), represented
// by the first three rows of each lifted rotation: arxiv.org/abs/2008.02737.
bool solveComponent(const Graph& graph, const ShonanOptions& options,
                    MatrixXd* rotations, ShonanReport* report) {
  size_t degree_sum = 0;
  size_t max_degree = 0;
  for (const auto& neighbors : graph) {
    degree_sum += neighbors.size();
    max_degree = std::max(max_degree, neighbors.size());
  }
  const double roundoff = 128.0 * std::numeric_limits<double>::epsilon() * max_degree;
  const double tolerance = options.gap_tolerance * (degree_sum / 2);
  MatrixXd x = *rotations;
  double best_cost = squaredError(graph, x);
  double lower_bound = 0.0;  // All edge costs are nonnegative.
  for (;;) {
    const int rank = static_cast<int>(x.cols());
    report->maximum_rank_used = std::max(report->maximum_rank_used, rank);
    // Avoid an eigensolve and even the first sweep on a consistent tree or
    // graph. Still refine small but measurable residuals in low-noise data.
    if (best_cost <= std::min(tolerance, 1e-24 * (degree_sum / 2))) break;

    minimize(graph, options.max_sweeps, &x);
    const double cost = squaredError(graph, x);
    MatrixXd candidate = rank == 3 ? x : roundToRotations(graph, x, options.max_sweeps);
    const double candidate_cost = rank == 3 ? cost : squaredError(graph, candidate);
    if (candidate_cost < best_cost) {
      best_cost = candidate_cost;
      rotations->swap(candidate);
    }
    if (best_cost - lower_bound <= tolerance) break;

    const auto certificate = certificateMatrix(graph, x);
    const double shift = 0.5 * tolerance / x.rows();
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> cholesky;
    cholesky.setShift(shift).compute(certificate);
    if (cholesky.info() == Eigen::Success) {
      lower_bound = std::max(lower_bound, cost - x.rows() * (shift + roundoff));
      if (best_cost - lower_bound <= tolerance) break;
      // A rounded solution near the tolerance boundary may still qualify
      // with a tighter lower bound from the unshifted certificate.
    }
    double lambda_lower_bound;
    VectorXd direction;
    if (!shonan_internal::minimumEigenpair(certificate, shift, &lambda_lower_bound, &direction)) {
      return false;
    }
    lower_bound = std::max(lower_bound, cost + x.rows() * std::min(0.0, lambda_lower_bound));
    if (best_cost - lower_bound <= tolerance) break;
    if (rank >= options.max_rank || rank >= x.rows() || direction.size() == 0 ||
        !lift(graph, direction, cost, &x)) {
      break;  // Return the feasible SO(3) estimate and its remaining gap.
    }
  }
  report->certified = report->certified && best_cost - lower_bound <= tolerance;
  report->squared_error += best_cost;
  report->optimality_gap += std::max(0.0, best_cost - lower_bound);
  return true;
}

}  // namespace

bool RotationAveraging::averageRotations(const Measurements& measurements, size_t num_rotations,
                                         std::vector<Quaterniond>* rotations) {
  return averageRotations(measurements, num_rotations, rotations, ShonanOptions{}, nullptr);
}

bool RotationAveraging::averageRotations(const Measurements& measurements, size_t num_rotations,
                                         std::vector<Quaterniond>* rotations,
                                         const ShonanOptions& options, ShonanReport* report) {
  if (!rotations || options.max_rank < 3 || options.max_sweeps < 1 ||
      !std::isfinite(options.gap_tolerance) || options.gap_tolerance <= 0 ||
      num_rotations > std::numeric_limits<int>::max() / 3) {
    return false;
  }

  ShonanReport summary;
  Graph graph(num_rotations);
  for (const auto& [indices, q] : measurements) {
    const auto [i, j] = indices;
    if (i >= num_rotations || j >= num_rotations || !q.coeffs().allFinite()) return false;
    const double scale = q.coeffs().cwiseAbs().maxCoeff();
    if (scale == 0.0) return false;
    Quaterniond unit;
    unit.coeffs() = q.coeffs() / scale;
    const Matrix3d measured = unit.normalized().toRotationMatrix();
    if (i == j) {
      summary.squared_error += (Matrix3d::Identity() - measured).squaredNorm();
    } else {
      graph[i].push_back({j, measured.transpose()});
      graph[j].push_back({i, measured});
    }
  }

  std::vector<Quaterniond> result(num_rotations, Quaterniond::Identity());
  std::vector<bool> visited(num_rotations, false);
  std::vector<size_t> local_index(num_rotations);
  for (size_t root = 0; root < num_rotations; ++root) {
    if (visited[root] || graph[root].empty()) continue;
    // A spanning tree gives a deterministic initial gauge and solves trees exactly.
    std::vector<size_t> nodes{root};
    visited[root] = true;
    for (size_t k = 0; k < nodes.size(); ++k) {
      const size_t i = nodes[k];
      for (const auto& edge : graph[i]) {
        if (visited[edge.node]) continue;
        visited[edge.node] = true;
        result[edge.node] = (Quaterniond(edge.to_here).conjugate() * result[i]).normalized();
        nodes.push_back(edge.node);
      }
    }

    std::sort(nodes.begin(), nodes.end());
    Graph component(nodes.size());
    MatrixXd frames(3 * nodes.size(), 3);
    for (size_t i = 0; i < nodes.size(); ++i) local_index[nodes[i]] = i;
    for (size_t i = 0; i < nodes.size(); ++i) {
      frames.middleRows<3>(3 * i) = result[nodes[i]].toRotationMatrix();
      for (const auto& edge : graph[nodes[i]]) {
        component[i].push_back({local_index[edge.node], edge.to_here});
      }
    }
    if (!solveComponent(component, options, &frames, &summary)) return false;

    const Matrix3d gauge = frames.topRows<3>().transpose();
    for (size_t i = 0; i < nodes.size(); ++i) {
      result[nodes[i]] = Quaterniond(frames.middleRows<3>(3 * i) * gauge).normalized();
    }
  }
  rotations->swap(result);
  if (report) *report = summary;
  return true;
}

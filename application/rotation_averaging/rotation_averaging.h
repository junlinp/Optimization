// Copyright (c) 2025 junlinp
// All rights reserved.

#ifndef ROTATION_AVERAGING_ROTATION_AVERAGING_H
#define ROTATION_AVERAGING_ROTATION_AVERAGING_H

#include <Eigen/Geometry>
#include <cstddef>
#include <map>
#include <utility>
#include <vector>

struct ShonanOptions {
  int max_rank = 6;        // Start at SO(3) and lift one dimension at a time.
  int max_sweeps = 1000;   // Block minimization sweeps per local solve.
  double gap_tolerance = 1e-8;  // Absolute optimality gap per non-self edge.
};

struct ShonanReport {
  bool certified = true;       // Every component met gap_tolerance.
  int maximum_rank_used = 0;
  double squared_error = 0.0;   // Sum of squared Frobenius residuals.
  double optimality_gap = 0.0;  // Upper bound on suboptimality of the SO(3) result.
};

class RotationAveraging {
 public:
  using Measurements = std::map<std::pair<std::size_t, std::size_t>, Eigen::Quaterniond>;

  // An edge (i,j) measures R_j R_i^T. Measurements have equal weight.
  // The lowest-index vertex in each connected component is fixed to identity.
  static bool averageRotations(const Measurements& measurements, std::size_t num_rotations,
                               std::vector<Eigen::Quaterniond>* rotations);

  // Both overloads return a feasible estimate on success. At a rank or
  // convergence limit the best SO(3) estimate and its gap are still useful;
  // inspect report->certified when a global optimality certificate is required.
  // Invalid input or numerical failure leaves rotations and report untouched.
  static bool averageRotations(const Measurements& measurements, std::size_t num_rotations,
                               std::vector<Eigen::Quaterniond>* rotations,
                               const ShonanOptions& options, ShonanReport* report);
};

#endif  // ROTATION_AVERAGING_ROTATION_AVERAGING_H

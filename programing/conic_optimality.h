#pragma once

#include <cmath>

// Shared stopping test in the original problem's units. Cone membership is
// checked separately by each backend before accepting an optimal iterate.
inline bool ConicOptimalityReached(double primal_residual, double dual_residual,
                                   double gap, double b_norm, double c_norm,
                                   double primal_objective, double tolerance) {
  return std::isfinite(primal_residual) && std::isfinite(dual_residual) &&
         std::isfinite(gap) && std::isfinite(primal_objective) &&
         primal_residual <= tolerance * (1.0 + b_norm) &&
         dual_residual <= tolerance * (1.0 + c_norm) &&
         gap >= 0.0 && gap <= tolerance * (1.0 + std::abs(primal_objective));
}

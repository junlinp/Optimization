#ifndef ROTATION_AVERAGING_SHONAN_CERTIFICATE_H
#define ROTATION_AVERAGING_SHONAN_CERTIFICATE_H

#include <Eigen/SparseCore>

namespace shonan_internal {

// Estimate lambda_min from below to numerical precision. If the matrix has
// detectable negative curvature, also return a unit descent direction.
bool minimumEigenpair(const Eigen::SparseMatrix<double>& certificate, double target_shift,
                      double* lower_bound, Eigen::VectorXd* direction);

}  // namespace shonan_internal

#endif  // ROTATION_AVERAGING_SHONAN_CERTIFICATE_H

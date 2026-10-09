#ifndef VIO_EQVIO_LIE_GROUPS_H_
#define VIO_EQVIO_LIE_GROUPS_H_
// The two groups EqVIO (van Goor & Mahony, arXiv:2205.01980) needs that
// Sophus does not ship: the extended special Euclidean group SE_2(3) for the
// navigation states, and the scaled orthogonal transforms SOT(3) for the
// landmarks. Both are hand-rolled here (see Sec. 3 of the paper for the
// definitions); SE(3) and SO(3) come from Sophus.
#include <Eigen/Dense>

#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

namespace vio {

using Vector9d = Eigen::Matrix<double, 9, 1>;

// Left Jacobian of SO(3): J_l(phi) = sum_k (phi^)^k / (k+1)!.
Eigen::Matrix3d SO3LeftJacobian(const Eigen::Vector3d& phi);
Eigen::Matrix3d SO3LeftJacobianInverse(const Eigen::Vector3d& phi);

// SE_2(3) element [[R, x, v], [0, 1, 0], [0, 0, 1]]. Tangent vectors are
// ordered (Omega, u, w): Omega the rotation, u the x-column, w the v-column.
struct SE23 {
  Sophus::SO3d R;
  Eigen::Vector3d x = Eigen::Vector3d::Zero();
  Eigen::Vector3d v = Eigen::Vector3d::Zero();

  SE23 operator*(const SE23& other) const;
  SE23 inverse() const;
  Sophus::SE3d pose() const { return Sophus::SE3d(R, x); }

  static SE23 exp(const Vector9d& xi);
  static Vector9d log(const SE23& X);
};

// SOT(3) = SO(3) x R_{>0}, Q = (R, c). Tangent vectors are (Omega, s) with
// c = e^s. Acts on R^3 \ {0} from the right by phi_V(Q, q) = c^-1 R^T q
// (paper eq. 17), so the product is componentwise: (R1 R2, c1 c2).
struct SOT3 {
  Sophus::SO3d R;
  double c = 1.0;

  SOT3 operator*(const SOT3& other) const { return {R * other.R, c * other.c}; }
  SOT3 inverse() const { return {R.inverse(), 1.0 / c}; }
  Eigen::Vector3d Act(const Eigen::Vector3d& q) const { return (R.inverse() * q) / c; }

  static SOT3 exp(const Eigen::Vector3d& omega, double s) {
    return {Sophus::SO3d::exp(omega), std::exp(s)};
  }
};

}  // namespace vio
#endif  // VIO_EQVIO_LIE_GROUPS_H_

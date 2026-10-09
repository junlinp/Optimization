#include "eqvio_lie_groups.h"

#include <cmath>

namespace vio {

Eigen::Matrix3d SO3LeftJacobian(const Eigen::Vector3d& phi) {
  const double theta = phi.norm();
  const Eigen::Matrix3d W = Sophus::SO3d::hat(phi);
  if (theta < 1e-8) return Eigen::Matrix3d::Identity() + 0.5 * W;
  const double t2 = theta * theta;
  return Eigen::Matrix3d::Identity() + ((1.0 - std::cos(theta)) / t2) * W +
         ((theta - std::sin(theta)) / (t2 * theta)) * W * W;
}

Eigen::Matrix3d SO3LeftJacobianInverse(const Eigen::Vector3d& phi) {
  const double theta = phi.norm();
  const Eigen::Matrix3d W = Sophus::SO3d::hat(phi);
  if (theta < 1e-8) return Eigen::Matrix3d::Identity() - 0.5 * W;
  const double t2 = theta * theta;
  const double coeff = 1.0 / t2 - (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta));
  return Eigen::Matrix3d::Identity() - 0.5 * W + coeff * W * W;
}

SE23 SE23::operator*(const SE23& other) const {
  return {R * other.R, x + R * other.x, v + R * other.v};
}

SE23 SE23::inverse() const {
  const Sophus::SO3d Rinv = R.inverse();
  return {Rinv, -(Rinv * x), -(Rinv * v)};
}

// The x- and v-columns are two independent copies of SE(3)'s translation, so
// each picks up the same left Jacobian of the rotation part.
SE23 SE23::exp(const Vector9d& xi) {
  const Eigen::Vector3d phi = xi.segment<3>(0);
  const Eigen::Matrix3d J = SO3LeftJacobian(phi);
  return {Sophus::SO3d::exp(phi), J * xi.segment<3>(3), J * xi.segment<3>(6)};
}

Vector9d SE23::log(const SE23& X) {
  const Eigen::Vector3d phi = X.R.log();
  const Eigen::Matrix3d Jinv = SO3LeftJacobianInverse(phi);
  Vector9d xi;
  xi << phi, Jinv * X.x, Jinv * X.v;
  return xi;
}

}  // namespace vio

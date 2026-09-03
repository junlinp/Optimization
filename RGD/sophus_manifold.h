#ifndef RGD_SOPHUS_MANIFOLD_H_
#define RGD_SOPHUS_MANIFOLD_H_
#include <Eigen/Dense>

#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

// Sophus-backed manifolds satisfying the same static contract as the
// hand-rolled classes in manifold.h, so they drop into ProductManifold,
// GradientChecker and both rgd() overloads unchanged.
//
// The difference from RotationMatrixManifold is the tangent space. The
// hand-rolled version carries tangent vectors in the 9-dimensional ambient
// space; here the tangent space is the Lie algebra itself, so SO(3) gets 3
// coordinates and SE(3) gets 6. A Riemannian Hessian built on these is
// therefore full rank instead of carrying a 6-dimensional degenerate subspace.
//
// The ambient layout is unchanged (column-major 3x3 rotation, then translation
// for SE(3)) so these are directly comparable against the hand-rolled classes
// on identical inputs.

class SophusSO3Manifold {
 public:
  constexpr static int AmbientSpaceSize = 9;
  constexpr static int TangentSpaceSize = 3;
  using AmbientSpaceVector = Eigen::Matrix<double, AmbientSpaceSize, 1>;
  using TangentSpaceVector = Eigen::Matrix<double, TangentSpaceSize, 1>;
  using GeneralJacobianVector = Eigen::Matrix<double, AmbientSpaceSize, 1>;

  SophusSO3Manifold() = delete;

  static AmbientSpaceVector IdentityElement() {
    return Flatten(Eigen::Matrix3d::Identity());
  }

  static AmbientSpaceVector RandomElement() {
    return Flatten(Sophus::SO3d::exp(Eigen::Vector3d::Random()).matrix());
  }

  // R <- R * exp(v^), the exponential retraction. Exact on the group rather
  // than the QR/Cayley approximations used by RotationMatrixManifold.
  static AmbientSpaceVector Retraction(const AmbientSpaceVector &x,
                                       const TangentSpaceVector &v) {
    return Flatten(Unflatten(x) * Sophus::SO3d::exp(v).matrix());
  }

  // Pull the ambient gradient back to so(3) coordinates.
  //
  // For the retraction above, d/dv f(X exp(v^))|_0 = tr(M^T v^) with
  // M = X^T G, and that equals <vee(M - M^T), v>. This is the minimal-
  // coordinate form of the X * skew(X^T U) projection in
  // so3_cost_function_interface.h.
  static TangentSpaceVector Project(
      const AmbientSpaceVector &x,
      const GeneralJacobianVector &general_gradient) {
    Eigen::Matrix3d M = Unflatten(x).transpose() * Unflatten(general_gradient);
    return Sophus::SO3d::vee(M - M.transpose());
  }

  // Every 3-vector is a valid tangent coordinate, so this cannot fail. It
  // exists because GradientChecker asserts on it.
  static bool IsTangentSpaceVector(const AmbientSpaceVector & /*x*/,
                                   const TangentSpaceVector & /*v*/) {
    return true;
  }

  // Lift a tangent coordinate into the 9-dimensional ambient representation
  // used by RotationMatrixManifold, for comparing the two side by side.
  static AmbientSpaceVector ToAmbient(const AmbientSpaceVector &x,
                                      const TangentSpaceVector &v) {
    return Flatten(Unflatten(x) * Sophus::SO3d::hat(v));
  }

  static Eigen::Matrix3d Unflatten(const AmbientSpaceVector &x) {
    return Eigen::Map<const Eigen::Matrix3d>(x.data());
  }

  static AmbientSpaceVector Flatten(const Eigen::Matrix3d &m) {
    return Eigen::Map<const AmbientSpaceVector>(m.data());
  }
};

// Real SE(3), not the SO(3) x R^3 product that manifold.h calls
// SepecialEuclideanManifold. The retraction couples rotation and translation
// through the left Jacobian, so a pure-rotation tangent moves the translation
// too.
//
// Ambient layout matches SepecialEuclideanManifold: 9 rotation entries
// (column-major) then 3 translation entries. Tangent ordering follows that
// same convention, rotation first then translation, which is the reverse of
// Sophus's native [upsilon, omega] ordering; the conversion is done here.
class SophusSE3Manifold {
 public:
  constexpr static int AmbientSpaceSize = 12;
  constexpr static int TangentSpaceSize = 6;
  using AmbientSpaceVector = Eigen::Matrix<double, AmbientSpaceSize, 1>;
  using TangentSpaceVector = Eigen::Matrix<double, TangentSpaceSize, 1>;
  using GeneralJacobianVector = Eigen::Matrix<double, AmbientSpaceSize, 1>;

  SophusSE3Manifold() = delete;

  static AmbientSpaceVector IdentityElement() { return Flatten(Sophus::SE3d()); }

  static AmbientSpaceVector RandomElement() {
    return Flatten(Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d::Random()),
                                Eigen::Vector3d::Random()));
  }

  static AmbientSpaceVector Retraction(const AmbientSpaceVector &x,
                                       const TangentSpaceVector &v) {
    return Flatten(Unflatten(x) * Sophus::SE3d::exp(ToSophusTangent(v)));
  }

  static TangentSpaceVector Project(
      const AmbientSpaceVector &x,
      const GeneralJacobianVector &general_gradient) {
    const Eigen::Matrix3d R = Unflatten(x).rotationMatrix();
    Eigen::Map<const Eigen::Matrix3d> G_r(general_gradient.data());
    Eigen::Map<const Eigen::Vector3d> G_t(general_gradient.data() + 9);

    // d/dv f(X exp(v^)) splits into a rotation part matching SophusSO3Manifold
    // and a translation part R^T G_t, because exp moves the translation by
    // R * upsilon to first order.
    const Eigen::Matrix3d M = R.transpose() * G_r;
    TangentSpaceVector res;
    res.head<3>() = Sophus::SO3d::vee(M - M.transpose());
    res.tail<3>() = R.transpose() * G_t;
    return res;
  }

  static bool IsTangentSpaceVector(const AmbientSpaceVector & /*x*/,
                                   const TangentSpaceVector & /*v*/) {
    return true;
  }

  static Sophus::SE3d Unflatten(const AmbientSpaceVector &x) {
    Eigen::Map<const Eigen::Matrix3d> R(x.data());
    Eigen::Map<const Eigen::Vector3d> t(x.data() + 9);
    return Sophus::SE3d(Sophus::SO3d::fitToSO3(R), t);
  }

  static AmbientSpaceVector Flatten(const Sophus::SE3d &pose) {
    AmbientSpaceVector res;
    const Eigen::Matrix3d R = pose.rotationMatrix();
    res.head<9>() = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(R.data());
    res.tail<3>() = pose.translation();
    return res;
  }

 private:
  // [rotation, translation] -> Sophus's [upsilon, omega].
  static Sophus::SE3d::Tangent ToSophusTangent(const TangentSpaceVector &v) {
    Sophus::SE3d::Tangent res;
    res.head<3>() = v.tail<3>();
    res.tail<3>() = v.head<3>();
    return res;
  }
};

#endif  // RGD_SOPHUS_MANIFOLD_H_

#include "sophus_manifold.h"

#include "gtest/gtest.h"
#include "manifold.h"
#include "rgd.h"
#include "so3_cost_function_interface.h"

namespace {

// f(R) = 0.5 * ||R - A||_F^2, whose ambient gradient is R - A.
class FrobeniusCost {
 public:
  explicit FrobeniusCost(const Eigen::Matrix3d &target) : target_(target) {}

  double Evaluate(const Eigen::Matrix<double, 9, 1> &x) const {
    return 0.5 * (SophusSO3Manifold::Unflatten(x) - target_).squaredNorm();
  }

  Eigen::Matrix<double, 9, 1> AmbientGradient(
      const Eigen::Matrix<double, 9, 1> &x) const {
    return SophusSO3Manifold::Flatten(SophusSO3Manifold::Unflatten(x) - target_);
  }

 private:
  Eigen::Matrix3d target_;
};

Eigen::Matrix3d SampleRotation() {
  return Sophus::SO3d::exp(Eigen::Vector3d(0.3, -0.7, 1.1)).matrix();
}

Eigen::Matrix3d SampleTarget() {
  return Sophus::SO3d::exp(Eigen::Vector3d(-0.4, 0.2, 0.9)).matrix();
}

// The projection defined by so3_cost_function_interface.h, X * skew(X^T U),
// reached through the public SO3Manifold wrapper.
Eigen::Matrix<double, 9, 1> SO3ManifoldProject(
    const Eigen::Matrix<double, 9, 1> &x,
    const Eigen::Matrix<double, 9, 1> &g) {
  return SO3Manifold::Project(x, g);
}

}  // namespace

// A retraction has to land back on the group. RotationMatrixManifold's Cayley
// transform satisfies this too; this is the baseline both must clear.
TEST(SophusSO3Manifold, RetractionStaysOnTheGroup) {
  const Eigen::Matrix<double, 9, 1> x =
      SophusSO3Manifold::Flatten(SampleRotation());
  const Eigen::Vector3d v(0.25, -0.4, 0.15);

  const Eigen::Matrix3d R =
      SophusSO3Manifold::Unflatten(SophusSO3Manifold::Retraction(x, v));

  EXPECT_LT((R.transpose() * R - Eigen::Matrix3d::Identity()).norm(), 1e-12);
  EXPECT_NEAR(R.determinant(), 1.0, 1e-12);
}

// exp is exact where the hand-rolled Cayley retraction is only a first-order
// approximation, so the two agree to O(t^2) and no better.
TEST(SophusSO3Manifold, RetractionIsFirstOrderConsistentWithCayley) {
  const Eigen::Matrix3d X = SampleRotation();
  const Eigen::Matrix<double, 9, 1> x = SophusSO3Manifold::Flatten(X);
  const Eigen::Vector3d omega(0.6, -0.2, 0.35);

  const double t = 1e-4;
  const Eigen::Matrix3d sophus_step =
      SophusSO3Manifold::Unflatten(SophusSO3Manifold::Retraction(x, t * omega));

  // Same step expressed in the 9-dimensional ambient tangent representation.
  const Eigen::Matrix<double, 9, 1> ambient =
      SophusSO3Manifold::ToAmbient(x, t * omega);
  const Eigen::Matrix3d cayley_step = SophusSO3Manifold::Unflatten(
      RotationMatrixManifold::Retraction(x, ambient));

  EXPECT_LT((sophus_step - cayley_step).norm(), 1e-7);
  EXPECT_GT((sophus_step - X).norm(), 1e-6);
}

// The defining property of a Riemannian gradient: it must reproduce the ambient
// gradient's action on every tangent direction. This is what separates the two
// conflicting SO(3) projections in this package, with no finite differences
// involved.
TEST(SophusSO3Manifold, ProjectionReproducesAmbientGradientOnTangents) {
  const Eigen::Matrix3d X = SampleRotation();
  const Eigen::Matrix<double, 9, 1> x = SophusSO3Manifold::Flatten(X);
  const FrobeniusCost cost(SampleTarget());
  const Eigen::Matrix<double, 9, 1> G = cost.AmbientGradient(x);

  const Eigen::Vector3d grad = SophusSO3Manifold::Project(x, G);

  for (int i = 0; i < 3; ++i) {
    const Eigen::Vector3d e = Eigen::Vector3d::Unit(i);
    // Tangent direction at X, in ambient coordinates.
    const Eigen::Matrix<double, 9, 1> xi = SophusSO3Manifold::ToAmbient(x, e);
    EXPECT_NEAR(grad.dot(e), G.dot(xi), 1e-12)
        << "Sophus projection failed direction " << i;
  }
}

// The same property, applied to both hand-rolled projections. These disagreed
// until RotationMatrixManifold::Project was changed to resolve X^T V rather
// than V; before that fix rmm_worst here was on the order of 1e-1.
TEST(RotationMatrixManifold, ProjectionAgreesWithSO3Manifold) {
  const Eigen::Matrix3d X = SampleRotation();
  const Eigen::Matrix<double, 9, 1> x = SophusSO3Manifold::Flatten(X);
  const FrobeniusCost cost(SampleTarget());
  const Eigen::Matrix<double, 9, 1> G = cost.AmbientGradient(x);

  const Eigen::Matrix<double, 9, 1> so3_grad = SO3ManifoldProject(x, G);
  const Eigen::Matrix<double, 9, 1> rmm_grad =
      RotationMatrixManifold::Project(x, G);

  double so3_worst = 0.0;
  double rmm_worst = 0.0;
  for (int i = 0; i < 3; ++i) {
    const Eigen::Matrix<double, 9, 1> xi =
        SophusSO3Manifold::ToAmbient(x, Eigen::Vector3d::Unit(i));
    so3_worst = std::max(so3_worst, std::abs(so3_grad.dot(xi) - G.dot(xi)));
    rmm_worst = std::max(rmm_worst, std::abs(rmm_grad.dot(xi) - G.dot(xi)));
  }

  // Both must now reproduce the ambient gradient on every tangent direction.
  EXPECT_LT(so3_worst, 1e-12);
  EXPECT_LT(rmm_worst, 1e-12);

  // And they must agree with each other, not merely each pass in isolation.
  EXPECT_LT((so3_grad - rmm_grad).norm(), 1e-12);
}

// Sophus's projection agrees with the correct hand-rolled one once lifted back
// into the ambient representation, which is the ground-truth cross-check.
TEST(SophusSO3Manifold, AgreesWithSO3ManifoldProjection) {
  const Eigen::Matrix3d X = SampleRotation();
  const Eigen::Matrix<double, 9, 1> x = SophusSO3Manifold::Flatten(X);
  const FrobeniusCost cost(SampleTarget());
  const Eigen::Matrix<double, 9, 1> G = cost.AmbientGradient(x);

  // SO3Manifold uses X * skew(X^T G); the minimal form drops the factor of two
  // that vee(M - M^T) carries relative to skew(M) = (M - M^T) / 2.
  const Eigen::Matrix<double, 9, 1> lifted = SophusSO3Manifold::ToAmbient(
      x, 0.5 * SophusSO3Manifold::Project(x, G));

  EXPECT_LT((lifted - SO3ManifoldProject(x, G)).norm(), 1e-12);
}

// First-order model check against the exponential retraction: the residual has
// to fall off as t^2.
TEST(SophusSO3Manifold, GradientMatchesFiniteDifferences) {
  const Eigen::Matrix<double, 9, 1> x =
      SophusSO3Manifold::Flatten(SampleRotation());
  const FrobeniusCost cost(SampleTarget());

  const Eigen::Vector3d grad =
      SophusSO3Manifold::Project(x, cost.AmbientGradient(x));
  const Eigen::Vector3d v = Eigen::Vector3d(0.5, -0.8, 0.3).normalized();
  const double f0 = cost.Evaluate(x);
  const double directional = grad.dot(v);

  double previous = 0.0;
  for (double t : {1e-2, 1e-3, 1e-4}) {
    const double ft = cost.Evaluate(SophusSO3Manifold::Retraction(x, t * v));
    const double residual = std::abs(ft - f0 - t * directional);
    EXPECT_LT(residual, 10.0 * t * t) << "at t = " << t;
    if (previous > 0.0) {
      // Shrinking t by 10 should shrink a second-order residual by ~100.
      EXPECT_LT(residual, previous / 50.0) << "at t = " << t;
    }
    previous = residual;
  }
}

// SE(3) is not the SO(3) x R^3 product that manifold.h names
// SepecialEuclideanManifold: the translation update is rotated into the body
// frame and coupled to the rotation through the left Jacobian.
TEST(SophusSE3Manifold, DiffersFromTheProductManifold) {
  const Eigen::Matrix3d R = SampleRotation();
  const Eigen::Vector3d t(1.5, -2.0, 0.75);

  Eigen::Matrix<double, 12, 1> x;
  x.head<9>() = SophusSO3Manifold::Flatten(R);
  x.tail<3>() = t;

  const Eigen::Vector3d omega(0.2, 0.1, -0.3);
  const Eigen::Vector3d upsilon(0.4, -0.25, 0.6);

  Eigen::Matrix<double, 6, 1> minimal;
  minimal.head<3>() = omega;
  minimal.tail<3>() = upsilon;
  const Eigen::Matrix<double, 12, 1> se3_next =
      SophusSE3Manifold::Retraction(x, minimal);

  // Same motion under the product manifold: rotation retracted on its own,
  // translation simply added in the world frame.
  Eigen::Matrix<double, 12, 1> product_tangent;
  product_tangent.head<9>() = SophusSO3Manifold::ToAmbient(
      SophusSO3Manifold::Flatten(R), omega);
  product_tangent.tail<3>() = upsilon;
  const Eigen::Matrix<double, 12, 1> product_next =
      SepecialEuclideanManifold::Retraction(x, product_tangent);

  EXPECT_GT((se3_next.tail<3>() - product_next.tail<3>()).norm(), 1e-3);

  // SE(3)'s translation is exactly t + R * V(omega) * upsilon.
  const Eigen::Vector3d expected =
      t + R * Sophus::SE3d::exp((Eigen::Matrix<double, 6, 1>() << upsilon, omega)
                                    .finished())
                  .translation();
  EXPECT_LT((se3_next.tail<3>() - expected).norm(), 1e-12);
}

TEST(SophusSE3Manifold, ProjectionReproducesAmbientGradientOnTangents) {
  const Eigen::Matrix3d R = SampleRotation();
  const Eigen::Vector3d t(0.5, 1.25, -0.75);

  Eigen::Matrix<double, 12, 1> x;
  x.head<9>() = SophusSO3Manifold::Flatten(R);
  x.tail<3>() = t;

  // Arbitrary but fixed ambient gradient.
  Eigen::Matrix<double, 12, 1> G;
  G << 0.3, -0.2, 0.9, 1.1, -0.4, 0.25, 0.7, 0.15, -0.6, 0.45, -0.85, 0.2;

  const Eigen::Matrix<double, 6, 1> grad = SophusSE3Manifold::Project(x, G);

  const double h = 1e-6;
  for (int i = 0; i < 6; ++i) {
    const Eigen::Matrix<double, 6, 1> e = Eigen::Matrix<double, 6, 1>::Unit(i);
    // Ambient velocity of the retraction along e, by central difference.
    const Eigen::Matrix<double, 12, 1> forward =
        SophusSE3Manifold::Retraction(x, h * e);
    const Eigen::Matrix<double, 12, 1> backward =
        SophusSE3Manifold::Retraction(x, -h * e);
    const Eigen::Matrix<double, 12, 1> velocity = (forward - backward) / (2 * h);

    EXPECT_NEAR(grad(i), G.dot(velocity), 1e-6)
        << "SE(3) projection failed direction " << i;
  }
}

namespace {

// Point-cloud registration on SE(3): recover T from correspondences
// q_i = T_true * p_i. Ambient layout is the 12-vector SophusSE3Manifold uses,
// and the tangent space is the 6-dimensional Lie algebra, so this also
// exercises rgd() with AmbientSpaceSize != TangentSpaceSize.
class Se3RegistrationCost : public RGDFirstOrderInterface {
 public:
  Se3RegistrationCost(const Sophus::SE3d &truth, int point_count) {
    for (int i = 0; i < point_count; ++i) {
      const Eigen::Vector3d p(std::sin(0.7 * i + 0.2), std::cos(1.3 * i),
                              0.4 * i - 1.0);
      source_.push_back(p);
      target_.push_back(truth * p);
    }
  }

  double Evaluate(const Eigen::VectorXd &x) const override {
    const Sophus::SE3d pose = SophusSE3Manifold::Unflatten(x.head<12>());
    double sum = 0.0;
    for (size_t i = 0; i < source_.size(); ++i) {
      sum += (pose * source_[i] - target_[i]).squaredNorm();
    }
    return 0.5 * sum;
  }

  Eigen::VectorXd Jacobian(const Eigen::VectorXd &x) const override {
    Eigen::Map<const Eigen::Matrix3d> R(x.data());
    Eigen::Map<const Eigen::Vector3d> t(x.data() + 9);

    Eigen::Matrix3d dR = Eigen::Matrix3d::Zero();
    Eigen::Vector3d dt = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < source_.size(); ++i) {
      const Eigen::Vector3d residual = R * source_[i] + t - target_[i];
      dR += residual * source_[i].transpose();
      dt += residual;
    }

    Eigen::Matrix<double, 12, 1> g;
    g.head<9>() = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(dR.data());
    g.tail<3>() = dt;
    return g;
  }

  Eigen::VectorXd ProjectExtendedGradientToTangentSpace(
      const Eigen::VectorXd &x,
      const Eigen::VectorXd &general_gradient) const override {
    return SophusSE3Manifold::Project(x.head<12>(), general_gradient.head<12>());
  }

  Eigen::VectorXd Move(const Eigen::VectorXd &x,
                       const Eigen::VectorXd &direction) const override {
    return SophusSE3Manifold::Retraction(x.head<12>(), direction.head<6>());
  }

 private:
  std::vector<Eigen::Vector3d> source_;
  std::vector<Eigen::Vector3d> target_;
};

}  // namespace

// End-to-end check that the adapter drives the existing solver unchanged.
TEST(SophusSE3Manifold, DrivesRgdToTheKnownPose) {
  const Sophus::SE3d truth(Sophus::SO3d::exp(Eigen::Vector3d(0.2, -0.35, 0.5)),
                           Eigen::Vector3d(1.2, -0.8, 2.5));
  auto cost = std::make_shared<Se3RegistrationCost>(truth, 24);

  Eigen::VectorXd x = SophusSE3Manifold::IdentityElement();
  const double initial = cost->Evaluate(x);

  rgd(cost, &x);

  EXPECT_LT(cost->Evaluate(x), 1e-9);
  EXPECT_LT(cost->Evaluate(x), initial);

  const Sophus::SE3d recovered = SophusSE3Manifold::Unflatten(x.head<12>());
  EXPECT_LT((recovered.inverse() * truth).log().norm(), 1e-5);
}

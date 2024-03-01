#ifndef RGD_MANIFOLD_H_
#define RGD_MANIFOLD_H_
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>

template <int ambient_space_size, int tangent_space_size>
class Manifold {
 public:
  constexpr static int AmbientSpaceSize = ambient_space_size;
  constexpr static int TangentSpaceSize = tangent_space_size;
  using AmbientSpaceVector = Eigen::Matrix<double, ambient_space_size, 1>;
  using TangentSpaceVector = Eigen::Matrix<double, tangent_space_size, 1>;
  using GeneralJacobianVector = Eigen::Matrix<double, ambient_space_size, 1>;

  // y = Retraction(x + v)
  static Manifold Retraction(const AmbientSpaceVector &x, const TangentSpaceVector &v);

  // so we project the general gradient of f to the tangent sapce of manifold at
  // point x
  static TangentSpaceVector Project(
      const AmbientSpaceVector &x, const GeneralJacobianVector &general_gradient);
};

template <int SPACE_SIZE>
class EuclideanManifold {
 public:
  constexpr static int AmbientSpaceSize = SPACE_SIZE;
  constexpr static int TangentSpaceSize = SPACE_SIZE;
  using AmbientSpaceVector = Eigen::Matrix<double, AmbientSpaceSize, 1>;
  using TangentSpaceVector = Eigen::Matrix<double, TangentSpaceSize, 1>;
  using GeneralJacobianVector = Eigen::Matrix<double, AmbientSpaceSize, 1>;

  EuclideanManifold() = delete; 

  // y = Retraction(x + v)
  static AmbientSpaceVector Retraction(const AmbientSpaceVector &x,
                                      const TangentSpaceVector &v) {
    return x + v;
  }

  // so we project the general gradient of f to the tangent sapce of manifold at
  // point x
  static TangentSpaceVector Project(
      const AmbientSpaceVector& /*x*/,
      const GeneralJacobianVector &general_gradient) {
    return general_gradient;
  }
};

class RotationMatrixManifold {
public:
  constexpr static int AmbientSpaceSize = 9;
  constexpr static int TangentSpaceSize = 9;
  using AmbientSpaceVector = Eigen::Matrix<double, AmbientSpaceSize, 1>;
  using TangentSpaceVector = Eigen::Matrix<double, TangentSpaceSize, 1>;
  using GeneralJacobianVector = Eigen::Matrix<double, AmbientSpaceSize, 1>;

  RotationMatrixManifold() = delete;

  // y = Retraction(x + v)
  static AmbientSpaceVector Retraction(const AmbientSpaceVector &x,
                                       const TangentSpaceVector &v) {
    return QRDecomposition(x, v);
  }

  // so we project the general gradient of f to the tangent sapce of manifold
  // at point x
  static TangentSpaceVector Project(
      const AmbientSpaceVector &x,
      const GeneralJacobianVector &general_gradient) {

    Eigen::Map<const Eigen::Matrix3d> X(x.data());
    Eigen::Map<const Eigen::Matrix3d> V(general_gradient.data());

    auto SkewPart = [](const Eigen::Matrix3d &X) -> Eigen::Matrix3d {
      return 0.5 * (X - X.transpose());
    };

    Eigen::Matrix3d XTU = X.transpose() * V;

    Eigen::Matrix3d Q = X * SkewPart(XTU);
    return Eigen::Map<TangentSpaceVector>(Q.data());
  }

 private:
   
  static TangentSpaceVector CayleyTransformation(const AmbientSpaceVector&x, const TangentSpaceVector& v) {
    Eigen::Map<const Eigen::Matrix3d> X(x.data());
    Eigen::Map<const Eigen::Matrix3d> V(v.data());
    Eigen::Matrix3d Identity = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d W =
        (Identity - 0.5 * X * X.transpose()) * V * X.transpose() -
        X * V.transpose() * (Identity - 0.5 * X * X.transpose());
    
    Eigen::Matrix3d Q = (Identity - 0.5 * W).inverse() * (Identity + 0.5 * W) * X;
    return Eigen::Map<TangentSpaceVector>(Q.data());
  }

  static TangentSpaceVector QRDecomposition(const AmbientSpaceVector&x, const TangentSpaceVector& v) {
    Eigen::Map<const Eigen::Matrix3d> X(x.data());
    Eigen::Map<const Eigen::Matrix3d> V(v.data());

    auto QR_solver = (X + V).householderQr();
    Eigen::Matrix3d Q = QR_solver.householderQ();
    Eigen::Matrix3d R = Q.transpose() * (X + V);
    Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
    if (Q.determinant() < 0.0) {
      for (int i = 0; i < 3; i++) {
        if (R(i, i) < 0.0) {
          identity(i, i) = -1.0;
        }
      }
    }
    Q = Q * identity;
    return Eigen::Map<TangentSpaceVector>(Q.data());
  }
};

template <class Manifold_lhs, class Manifold_rhs>
class ProductManifold {
 public:
  constexpr static int AmbientSpaceSize =
      Manifold_lhs::AmbientSpaceSize + Manifold_rhs::AmbientSpaceSize;
  constexpr static int TangentSpaceSize =
      Manifold_lhs::TangentSpaceSize + Manifold_rhs::TangentSpaceSize;

  using AmbientSpaceVector = Eigen::Matrix<double, AmbientSpaceSize, 1>;
  using TangentSpaceVector = Eigen::Matrix<double, TangentSpaceSize, 1>;
  using GeneralJacobianVector = Eigen::Matrix<double, AmbientSpaceSize, 1>;

  ProductManifold() = delete;

  // y = Retraction(x + v)
  static AmbientSpaceVector Retraction(const AmbientSpaceVector&x,
                                    const TangentSpaceVector &v) {
    typename Manifold_lhs::TangentSpaceVector lhs_tangent =
        v.block(0, 0, Manifold_lhs::TangentSpaceSize, 1);

    typename Manifold_rhs::TangentSpaceVector rhs_tangent = v.block(
        Manifold_lhs::TangentSpaceSize, 0, Manifold_rhs::TangentSpaceSize, 1);

    typename Manifold_lhs::AmbientSpaceVector res_lhs =
        Manifold_lhs::Retraction(FirstPartAmbientVector(x), lhs_tangent);

    typename Manifold_rhs::AmbientSpaceVector res_rhs =
        Manifold_rhs::Retraction(SecondPartAmbientVector(x), rhs_tangent);
    AmbientSpaceVector res;
    res << res_lhs, res_rhs;
    return res;
  }

  // so we project the general gradient of f to the tangent sapce of manifold at
  // point x
  static TangentSpaceVector Project(
      const AmbientSpaceVector&x, const GeneralJacobianVector &general_gradient) {
    TangentSpaceVector res;

    res.block(0, 0, Manifold_lhs::TangentSpaceSize, 1) = Manifold_lhs::Project(
        FirstPartAmbientVector(x),
        general_gradient.block(0, 0, Manifold_lhs::TangentSpaceSize, 1));

    res.block(Manifold_lhs::TangentSpaceSize, 0, Manifold_rhs::TangentSpaceSize,
              1) =
        Manifold_rhs::Project(
            SecondPartAmbientVector(x),
            general_gradient.block(Manifold_lhs::TangentSpaceSize, 0,
                                   Manifold_rhs::TangentSpaceSize, 1));
    return res;
  }

 private:
  auto static FirstPartAmbientVector(const AmbientSpaceVector& ambient_space_vector) {
    return ambient_space_vector.block(0, 0, Manifold_lhs::AmbientSpaceSize, 1);
  }

  auto static SecondPartAmbientVector(const AmbientSpaceVector& ambient_space_vector) {
    return ambient_space_vector.block(Manifold_lhs::AmbientSpaceSize, 0,
                                      Manifold_rhs::AmbientSpaceSize, 1);
  }

};

using SepecialEuclideanManifold = ProductManifold<RotationMatrixManifold, EuclideanManifold<3>>;

template <class T>
using remove_cvrf_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <class T1, class T2>
auto CreateManifold(T1 &&manifold_lhs, T2 &&manifold_rhs) {
  return ProductManifold<remove_cvrf_t<T1>, remove_cvrf_t<T2>>(manifold_lhs,
                                                               manifold_rhs);
}

template <class T, class... ARGS>
auto CreateManifold(T &&manifold, ARGS... manifolds) {
  auto rhs = CreateManifold(manifolds...);
  return ProductManifold<remove_cvrf_t<T>, decltype(rhs)>(manifold, rhs);
}

#endif  // RGD_MANIFOLD_H_

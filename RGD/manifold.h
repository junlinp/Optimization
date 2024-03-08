#ifndef RGD_MANIFOLD_H_
#define RGD_MANIFOLD_H_
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <iostream>

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

  static AmbientSpaceVector IdentityElement() {
    return AmbientSpaceVector::Zero();
  }
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

template <int DIM>
class SphereManifold {
public:
  constexpr static int AmbientSpaceSize = DIM;
  constexpr static int TangentSpaceSize = DIM;
  using AmbientSpaceVector = Eigen::Matrix<double, DIM, 1>;
  using TangentSpaceVector = Eigen::Matrix<double, DIM, 1>;
  using GeneralJacobianVector = Eigen::Matrix<double, DIM, 1>;

  SphereManifold() = delete;

  static AmbientSpaceVector IdentityElement() {
    AmbientSpaceVector v;
    v.setRandom();
    v.normalized();
    return v;
  }

  static AmbientSpaceVector RandomElement() {
    AmbientSpaceVector v;
    v.setRandom();
    v.normalized();
    v << 0.0 ,0.707, 0.707;
    return v;
  }

  // y = Retraction(x + v)
  static AmbientSpaceVector Retraction(const AmbientSpaceVector &x,
                                       const TangentSpaceVector &v) {
    //return QRDecomposition(x, v);
    AmbientSpaceVector r = x + v;
    return r / r.norm();
    //return r / std::sqrt(1.0 + v.dot(v));
  }

  // so we project the general gradient of f to the tangent sapce of manifold
  // at point x
  static TangentSpaceVector Project(
      const AmbientSpaceVector &x,
      const GeneralJacobianVector &general_gradient) {
    return (Eigen::MatrixXd::Identity(DIM, DIM) -
           x * x.transpose()) * general_gradient;
  }

  static bool IsTangentSpaceVector(const AmbientSpaceVector& x, const TangentSpaceVector& v) {
    return x.dot(v) < 1e-5;
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

  static AmbientSpaceVector IdentityElement() {
    AmbientSpaceVector v;
    v << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    return v;
  }

  static AmbientSpaceVector RandomElement() {

    AmbientSpaceVector v;

    double theta = 3.0;
    v << 1.0, 0.0, 0.0, 0.0, std::cos(theta), std::sin(theta), 0.0, -std::sin(theta), std::cos(theta);
    return v;
  }

  // y = Retraction(x + v)
  static AmbientSpaceVector Retraction(const AmbientSpaceVector &x,
                                       const TangentSpaceVector &v) {
    //return QRDecomposition(x, v);
    Eigen::Map<const Eigen::Matrix3d> w(v.data());

    Eigen::Vector3d u;
    u << w(2, 1), w(0, 2), w(1, 0);

    double theta = u.norm();
    Eigen::Matrix3d w_hat = w / theta;

    Eigen::Map<const Eigen::Matrix3d> X(x.data());
    double coe1 = 0.0, coe2 = 0.0;
    if (theta < 1e-7) {
      coe1 = 1.0 - theta / 2.0 * theta / 3.0 +
             theta / 2.0 * theta / 3.0 * theta / 4.0 * theta / 5.0;
      coe2 = 0.5 - theta / 4 * theta / 6 +
             theta / 2.0 * theta / 5.0 * theta / 6.0 * theta / 7.0;
    } else {
      coe1 = std::sin(theta) / theta;
      coe2 = (1.0 - std::cos(theta)) / theta / theta ;
    }
    /*
    return (X * (Eigen::Matrix3d::Identity() + coe1 * w_hat +
                 (coe2 * (w_hat * w_hat))))
        .reshaped();
    */
    return CayleyTransformation(x, v);
  }

  // so we project the general gradient of f to the tangent sapce of manifold
  // at point x
  static TangentSpaceVector Project(
      const AmbientSpaceVector &x,
      const GeneralJacobianVector &general_gradient) {

    Eigen::Map<const Eigen::Matrix3d> X(x.data());
    Eigen::Map<const Eigen::Matrix3d> V(general_gradient.data());
    Eigen::Matrix3d e1, e2, e3;

    e1 << 0, -1, 0,
          1, 0, 0,
          0, 0, 0;
    e2 << 0, 0, 1,
          0, 0, 0,
          -1, 0, 0;
    e3 << 0, 0, 0,
          0, 0, -1,
          0, 1, 0;
    
    Eigen::Matrix3d tangent_e1 =  e1;
    Eigen::Matrix3d tangent_e2 =  e2;
    Eigen::Matrix3d tangent_e3 =  e3;

    auto InnerProduct = [](Eigen::Matrix3d a, Eigen::Matrix3d b) -> double {
      return (a.transpose() * b).trace();
    };
    Eigen::Matrix3d remind_V = V; 
    double v1 = InnerProduct(remind_V, tangent_e1) / (InnerProduct(tangent_e1, tangent_e1));
    remind_V = remind_V - v1 * tangent_e1;
    double v2 = InnerProduct(remind_V, tangent_e2) / (InnerProduct(tangent_e2, tangent_e2));
    remind_V = remind_V - v2 * tangent_e2;
    double v3 = InnerProduct(V, tangent_e3) / (InnerProduct(tangent_e3, tangent_e3));
    remind_V = remind_V - v3 * tangent_e3;
    

    Eigen::Matrix3d TxU = v1 * e1 + v2 * e2 + v3 * e3;

    Eigen::Matrix3d Q = X * TxU;
    //std::cout << "TxU + TxU.transpose() : " << TxU + TxU.transpose() << std::endl;
    return Eigen::Map<TangentSpaceVector>(Q.data());
  }
  static bool IsTangentSpaceVector(const AmbientSpaceVector& x, const TangentSpaceVector& v) {
    Eigen::Map<const Eigen::Matrix3d> U(v.data());
    Eigen::Map<const Eigen::Matrix3d> X(x.data());

    Eigen::Matrix3d R = X.transpose() * U;
    return (R + R.transpose()).array().square().sum() < 1e-5;
  }

 private:
   
  static TangentSpaceVector CayleyTransformation(const AmbientSpaceVector&x, const TangentSpaceVector& v) {
    Eigen::Map<const Eigen::Matrix3d> X(x.data());
    Eigen::Map<const Eigen::Matrix3d> V(v.data());
    Eigen::Matrix3d Identity = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d W =
        (Identity - 0.5 * X * X.transpose()) * V * X.transpose() -
        X * V.transpose() * (Identity - 0.5 * X * X.transpose());

    Eigen::Matrix3d invertible_matrix = (Identity -0.5 * W);
    Eigen::FullPivLU<Eigen::Matrix3d> solver(invertible_matrix);
    Eigen::Matrix3d P = solver.permutationP();
    Eigen::Matrix3d L = Eigen::Matrix3d::Identity();
    L.triangularView<Eigen::StrictlyLower>() = solver.matrixLU();
    Eigen::Matrix3d U = solver.matrixLU().triangularView<Eigen::Upper>();
    Eigen::Matrix3d Q = solver.permutationQ();

    Eigen::Matrix3d res = Q * U.inverse() * L.inverse() * P * (Identity + 0.5 * W) * X;
    return Eigen::Map<TangentSpaceVector>(res.data());
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

  static AmbientSpaceVector IdentityElement() {
    AmbientSpaceVector v;
    v << Manifold_lhs::IdentityElement(), Manifold_rhs::IdentityElement();
    return v;
  }

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

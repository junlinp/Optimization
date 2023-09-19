#ifndef RGD_MANIFOLD_H_
#define RGD_MANIFOLD_H_
#include <Eigen/Dense>

template<int ambient_space_size, int tangent_space_size>
class Manifold {
public:
  constexpr static int AmbientSpaceSize = ambient_space_size;
  constexpr static int TangentSpaceSize = tangent_space_size;
  using AmbientSpaceVector = Eigen::Matrix<double, ambient_space_size, 1>;
  using TangentSpaceVector = Eigen::Matrix<double, tangent_space_size, 1>;
  using GeneralJacobianVector = Eigen::Matrix<double, ambient_space_size, 1>;
  
  // y = Retraction(x + v)
  static Manifold Retraction(const Manifold&x,
                                        const TangentSpaceVector &v);

  // f is a real value function for manifold.
  // so we project the general gradient of f to the tangent sapce of manifold at
  // point x
  static TangentSpaceVector Project(
      const Manifold &x, const GeneralJacobianVector &general_gradient);
};

template<int ambient_space_size, int tangent_space_size>
class EuclideanManifold {

};

class RotationMatrixManifold : public Manifold<9, 9> {
public:
  // y = Retraction(x + v)
  AmbientSpaceVector Retraction(const AmbientSpaceVector &x,
                                        const TangentSpaceVector &v) override;


  // f is a real value function for manifold.
  // so we project the general gradient of f to the tangent sapce of manifold at
  // point x
  TangentSpaceVector
  Project(const AmbientSpaceVector &x,
          const GeneralJacobianVector &general_gradient) override;

private:
  AmbientSpaceVector vector_;
};

template <class Manifold_lhs, class Manifold_rhs>
class ProductManifold
    : public Manifold<
          Manifold_lhs::AmbientSpaceSize + Manifold_rhs::AmbientSpaceSize,
          Manifold_lhs::TangentSpaceSize + Manifold_rhs::TangentSpaceSize> {
public:
  ProductManifold();
  ProductManifold(const AmbientSpaceVector &ambient_vector)
      : vector_(ambient_vector) {}

  ProductManifold(const Manifold_lhs& manifold_lhs, const Manifold_rhs& manifold_rhs) : {
    vector_.head<Manifold_lhs::AmbientSpaceSize> = manifold_lhs.AmbientVector();
    vector_.tail<Manifold_rhs::AmbientSpaceSize> = manifold_rhs.AmbientVector();
  }

  // y = Retraction(x + v)
  static ProductManifold Retraction(const ProductManifold &x,
                             const TangentSpaceVector &v) {
    Eigen::Map<Manifold_lhs::AmbientSpaceVector> lhs(x.vector_.data());

    Eigen::Map<Manifold_rhs::AmbientSpaceVector> rhs(
        x.vector_.data() + Manifold_lhs::AmbientSpaceSize);
    Manifold_lhs lhs_monifold(lhs);
    Manifold_rhs rhs_monifold(rhs);
    Eigen::Map<const Manifold_lhs::TangentSpaceVector> lhs_tangent(v.data());
    Eigen::Map<const Manifold_rhs::TangentSpaceVector> rhs_tangent(
        v.data() + Manifold_lhs::TangentSpaceSize);
    Manifold_lhs res_lhs = Manifold_lhs::Retraction(lhs_monifold, lhs_tangent);
    Manifold_rhs res_rhs = Manifold_rhs::Retraction(rhs_monifold, rhs_tangent);
    return ProductManifold(lhs_monifold, rhs_monifold);
  }

  // f is a real value function for manifold.
  // so we project the general gradient of f to the tangent sapce of manifold at
  // point x
  static TangentSpaceVector Project(
      const ProductManifold &x, const GeneralJacobianVector &general_gradient) {

  }

 private:
  AmbientSpaceVector vector_;
};

#endif // RGD_MANIFOLD_H_

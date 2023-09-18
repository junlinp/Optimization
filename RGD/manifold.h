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
  virtual AmbientSpaceVector Retraction(const AmbientSpaceVector &x,
                                        const TangentSpaceVector &v) = 0;

  // f is a real value function for manifold.
  // so we project the general gradient of f to the tangent sapce of manifold at
  // point x
  virtual TangentSpaceVector
  Project(const AmbientSpaceVector &x,
          const GeneralJacobianVector &general_gradient) = 0;
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
  // y = Retraction(x + v)
  AmbientSpaceVector Retraction(const AmbientSpaceVector &x,
                                const TangentSpaceVector &v) override{
        AmbientSpaceVector res = x;
      Eigen::Map<Manifold_lhs::AmbientSpaceVector> lhs(res.data());
      Eigen::Map<Manifold_rhs::AmbientSpaceVector> rhs(
          res.data() + Manifold_lhs::AmbientSpaceSize);

     Eigen::Map<const Manifold_lhs::TangentSpaceVector> lhs_tangent(v.data());
     Eigen::Map<const Manifold_rhs::TangentSpaceVector> rhs_tangent(v.data() + Manifold_lhs::TangentSpaceSize);

     lhs = Manifold_lhs
      }

  // f is a real value function for manifold.
  // so we project the general gradient of f to the tangent sapce of manifold at
  // point x
  TangentSpaceVector
      Project(const AmbientSpaceVector &x,
              const GeneralJacobianVector &general_gradient) override;

private:
  AmbientSpaceVector vector_;
};

#endif // RGD_MANIFOLD_H_

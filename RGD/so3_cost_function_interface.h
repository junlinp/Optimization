#ifndef RGD_SO3_COST_FUNCTION_H_
#define RGD_SO3_COST_FUNCTION_H_
#include <iostream>
#include "Eigen/Eigen"
namespace {

Eigen::Matrix3d SkewPart(const Eigen::Matrix3d &X) {
  return 0.5 * (X.transpose() - X);
}

Eigen::Matrix3d MatrixProject(const Eigen::Matrix3d &X, const Eigen::Matrix3d &U) {
  return X * SkewPart(X.transpose() * U);
}

Eigen::Matrix3d MatrixRetraction(const Eigen::Matrix3d &X, const Eigen::Matrix3d &V) {
  Eigen::Matrix3d Q =  (X + V).householderQr().householderQ();
  Eigen::Matrix3d R = (X + V).householderQr().matrixQR();
  std::cout << "X : " << X << std::endl;
  std::cout << " V : " << V << std::endl;
  std::cout << "X + V : " << X + V << std::endl;
  std::cout  << "Q * R : " << Q * R << std::endl;
  Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
  for (int i = 0; i < 3; i++) {
    if (R(i, i) < 0.0) {
        identity(i, i) = -1.0;
    }
  }
  Q = identity * Q;
  std::cout << "Retraction Diff : " << (Q - X).array().sum() << std::endl;
  return Q;
}
}
class SO3Manifold {
    public:
    // column major
    using Vector = Eigen::Matrix<double, 9, 1>;

    using TangentVector = Vector;

    constexpr Vector Identity(); 
    
    static TangentVector Project(const Vector& x,const Vector& general_gradient) {
        Eigen::Map<const Eigen::Matrix3d> matrix_x(x.data());
        Eigen::Map<const Eigen::Matrix3d> matrix_gradient(general_gradient.data());
        std::cout << "Project X : " << matrix_x << std::endl; 
        std::cout << "Project gradient : " << matrix_gradient << std::endl;
        Eigen::Matrix3d tangent = MatrixProject(matrix_x, matrix_gradient);
        std::cout << "Project tangent : " << tangent << std::endl;
        Eigen::Map<TangentVector> res(tangent.data());
        return res;
    }

    static Vector Retraction(const Vector& x,const TangentVector& tangent_vector) {
        Eigen::Map<const Eigen::Matrix3d> matrix_x(x.data());
        Eigen::Map<const Eigen::Matrix3d> matrix_vector(tangent_vector.data());

        Eigen::Matrix3d temp = MatrixRetraction(matrix_x, matrix_vector);
        std::cout << "det : " << temp.determinant() << std::endl;
        Eigen::Map<Vector> res(temp.data());
        return res;
    }
     static bool CheckTangentVector(const Vector& x, const TangentVector& tangent_vector) {
        Eigen::Map<const Eigen::Matrix3d> matrix_x(x.data());
        Eigen::Map<const Eigen::Matrix3d> matrix_gradient(tangent_vector.data());

        Eigen::Matrix3d res = matrix_x * matrix_gradient - (matrix_x * matrix_gradient).transpose();
        
        return res.array().sum() < 1e-6;
     }
};

class SO3CostFunctionInterface {
public:
    virtual double Evaluate(const std::vector<SO3Manifold::Vector>& x) const = 0;
    virtual std::vector<SO3Manifold::Vector> Jacobian(const std::vector<SO3Manifold::Vector>& x) const = 0;
    virtual ~SO3CostFunctionInterface() = default;
};
#endif // RGD_SO3_COST_FUNCTION_H_
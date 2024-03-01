#ifndef RGB_RGD_COST_FUNCTION_INTERFACE_H_
#define RGB_RGD_COST_FUNCTION_INTERFACE_H_

#include "Eigen/Dense"

// real value function
class RGDFirstOrderInterface {
public:
    virtual double Evaluate(const Eigen::VectorXd& x) const = 0;

    virtual Eigen::VectorXd Jacobian(const Eigen::VectorXd& x) const = 0;

    virtual Eigen::VectorXd ProjectExtendedGradientToTangentSpace(
      const Eigen::VectorXd&x, const Eigen::VectorXd &general_gradient) const = 0;

    virtual Eigen::VectorXd Move(const Eigen::VectorXd& x,
                                 const Eigen::VectorXd& direction) const = 0;

    virtual ~RGDFirstOrderInterface() = default;
};
#endif  //  RGB_RGD_COST_FUNCTION_INTERFACE_H_
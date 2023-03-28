#ifndef RGD_SO3_COST_FUNCTION_H_
#define RGD_SO3_COST_FUNCTION_H_

#include "Eigen/Eigen"

class SO3CostFunctionInterface {
public:
    virtual double Evaluate(const std::vector<Eigen::Matrix3d>& x) const = 0;
    virtual std::vector<Eigen::Matrix3d> Jacobian(const std::vector<Eigen::Matrix3d>& x) const = 0;
    virtual ~SO3CostFunctionInterface() = default;
};
#endif // RGD_SO3_COST_FUNCTION_H_
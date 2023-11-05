#ifndef FIRST_ORDER_METHODS_FIRST_ORACLE_H_
#define FIRST_ORDER_METHODS_FIRST_ORACLE_H_
#include "Eigen/Dense"
class FirstOracle {
    public:
      virtual double fval(const Eigen::VectorXd &x0) = 0;
      virtual Eigen::VectorXd SubGradient(const Eigen::VectorXd &x0) = 0;
      virtual ~FirstOracle() = default;
};

#endif // FIRST_ORDER_METHODS_FIRST_ORACLE_H_

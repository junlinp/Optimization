#ifndef FIRST_ORDER_METHODS_GRADIENT_DESCENT_H_
#define FIRST_ORDER_METHODS_GRADIENT_DESCENT_H_
#include "first_oracle.h"

Eigen::VectorXd GradientDescnet(std::shared_ptr<FirstOracle> function, const Eigen::VectorXd& x0, double L);

Eigen::VectorXd  NesterovGradientDescnet(std::shared_ptr<FirstOracle> function, const Eigen::VectorXd& x0, double L);


Eigen::VectorXd  NesterovGradientDescnet2(std::shared_ptr<FirstOracle> function, const Eigen::VectorXd& x0, double L);

#endif //  FIRST_ORDER_METHODS_GRADIENT_DESCENT_H_

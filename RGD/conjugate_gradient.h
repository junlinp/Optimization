
#include "Eigen/Dense"

bool ConjugateGradient(const Eigen::MatrixXd&A, const Eigen::VectorXd& b, double tolerance, Eigen::VectorXd* solution);
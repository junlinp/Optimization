#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "problem.h"
#include "ceres/rotation.h"

void Evaluate(const Problem& problem, Eigen::VectorXd& error, Eigen::SparseMatrix<double>& jacobian);
void LM(Problem& problem);
template<class T>
void AngleAxisRotation(T* angle_axis, T* point, T* output) {
    const T theta2 = (angle_axis[0] * angle_axis[0]) + (angle_axis[1] * angle_axis[1]) + (angle_axis[2] * angle_axis[2]);
    const T theta = sqrt(theta2);

    const T costheta = cos(theta);
    const T sintheta = sin(theta);
    const T theta_inverse = T(1.0) / theta;
    const T w[3] = {angle_axis[0] * theta_inverse,
                    angle_axis[1] * theta_inverse,
                    angle_axis[2] * theta_inverse
    };
    const T w_cross_pt[3] = {
        w[1] * point[2] - w[2] * point[1],
        w[2] * point[0] - w[0] * point[2],
        w[0] * point[1] - w[1] * point[0]
    };

    const T tmp = (w[0] * point[0] + w[1] * point[1] + w[2] * point[2]) * (T(1.0) - costheta);
    output[0] = costheta * point[0] + sintheta * w_cross_pt[0] + w[0] * tmp; 
    output[1] = costheta * point[1] + sintheta * w_cross_pt[1] + w[1] * tmp; 
    output[2] = costheta * point[2] + sintheta * w_cross_pt[2] + w[2] * tmp; 
}
#include "Eigen/Dense"
#include "problem.h"
#include "../JET.h"

void Evaluate(const Problem& problem, Eigen::VectorXd& error, Eigen::MatrixXd& jacobian);

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

struct ProjectFunction {
    ProjectFunction(double u, double v) : u(u), v(v) {}
    double u, v;
    template<class T>
    bool operator()(T* camera_param, T* point, T* residual) const {
        T output_point[3];
        AngleAxisRotation(camera_param, point, output_point);
        output_point[0] += camera_param[3];
        output_point[1] += camera_param[4];
        output_point[2] += camera_param[5];

        output_point[0] /= -output_point[2];
        output_point[1] /= -output_point[2];
        T focal = camera_param[6];
        T K1 = camera_param[7];
        T K2 = camera_param[8];

        T p_norm_2 = output_point[0] * output_point[0] + output_point[1] * output_point[1];
        T distorsion = T(1.0) + K1 * p_norm_2 + K2 * p_norm_2 * p_norm_2;
        residual[0] = T(u) - focal * distorsion * output_point[0];
        residual[1] = T(v) - focal * distorsion * output_point[1];

        return true;
    }
};
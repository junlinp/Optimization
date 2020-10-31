#include "evaluate.h"
#include "../JET.h"
#include "iostream"


struct ProjectFunction {
    ProjectFunction(double u, double v) : u(u), v(v) {}
    double u, v;
    template<class T>
    bool operator()(T* camera_param, T* point, T* residual) const {
        T output_point[3];
        AngleAxisRotation(camera_param, point, output_poiont);
        output_point[0] += camera_param[3];
        output_point[1] += camera_param[4];
        output_point[2] += camera_param[5];

        output_point[0] /= -output_point[2];
        output_point[1] /= -output_point[2];
        T focal = camera_param[6];
        T K1 = camera_param[7];
        T K8 = camera_param[8];

        T p_norm_2 = output_point[0] * output_point[0] + output_point[1] * output_point[1];
        T distorsion = T(1.0) + K1 * p_norm_2 + K2 * p_norm_2 * p_norm_2;
        residual[0] = T(u) - focal * distorsion * output_poiont[0];
        residual[1] = T(v) - focal * distorsion * output_poiont[1];

        return true;
    }
};
void Evaluate(const Problem& problem, Eigen::VectorXd& error, Eigen::MatrixXd& jacobian) {
    for(const auto& pair : problem.observations_) {
        size_t camera_ids = pair.first.first;
        size_t point_ids = pair.first.second;
        Observation o = pair.second;
        ProjectFunction pf(o(0), o(1));
        Jet<9 + 3> X[12];
        CameraParam c = problem.cameras_.at(camera_ids);
        Landmark p = problem.points_.at(point_ids);
        X[0] = Jet<12>(c.params[0], 0);
        X[1] = Jet<12>(c.params[1], 1);
        X[2] = Jet<12>(c.params[2], 2);
        X[3] = Jet<12>(c.params[3], 3);
        X[4] = Jet<12>(c.params[4], 4);
        X[5] = Jet<12>(c.params[5], 5);
        X[6] = Jet<12>(c.params[6], 6);
        X[7] = Jet<12>(c.params[7], 7);
        X[8] = Jet<12>(c.params[8], 8);
        X[9] = Jet<12>(p(0), 9);
        X[10] = Jet<12>(p(1), 10);
        X[11] = Jet<12>(p(2), 11);
        Jet<12> residual[2];
        pf(X, X + 9, residual);
        std::cout << "Residual : " << residual[0].value() << " , " << residual[1].value() << std::endl;
    }
}


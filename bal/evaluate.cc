#include "evaluate.h"
#include "iostream"



void Evaluate(const Problem& problem, Eigen::VectorXd& error, Eigen::MatrixXd& jacobian) {
    size_t observation_size = problem.observations_.size();
    size_t camera_size = problem.cameras_.size();
    size_t point_size = problem.points_.size();
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
    error = Eigen::VectorXd(2 * observation_size);
    jacobian = Eigen::MatrixXd(2 * observation_size, camera_size * 9 + point_size * 3);
}


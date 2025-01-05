// Copyright (c) [2025] junlinp. All rights reserved.
#ifndef RGD_TRUST_REGION_SUBPROBLEM_H_
#define RGD_TRUST_REGION_SUBPROBLEM_H_
#include "Eigen/Dense"

bool TrustRegionSubProblem(const Eigen::MatrixXd &A, const Eigen::VectorXd &B,
                           double trust_region_radius,
                           Eigen::VectorXd *initial_solution);

#endif // RGD_TRUST_REGION_SUBPROBLEM_H_

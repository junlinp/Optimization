// Copyright (c) 2025 junlinp
// All rights reserved.

#ifndef ROTATION_AVERAGING_ROTATION_AVERAGING_H
#define ROTATION_AVERAGING_ROTATION_AVERAGING_H

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <utility>

class RotationAveraging {
public:
  static Eigen::Quaterniond averageRotations(
      const std::map<std::pair<int, int>, Eigen::Quaterniond> &relative_rotations,
      size_t num_of_rotations, std::vector<Eigen::Quaterniond> *rotations);
};

#endif // ROTATION_AVERAGING_ROTATION_AVERAGING_H




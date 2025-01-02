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
  static bool averageRotations(
      const std::map<std::pair<size_t, size_t>, Eigen::Quaterniond> &relative_rotations,
      size_t num_of_rotations, std::vector<Eigen::Quaterniond> *rotations);
};

#endif // ROTATION_AVERAGING_ROTATION_AVERAGING_H




#include <gtest/gtest.h>
#include <Eigen/Dense>

TEST(RotationAveragingTest, BasicFunctionality) {
    // Setup test data
    size_t n = 128;

    std::vector<Eigen::Quaterniond> rotations;

    for (size_t i = 0; i < n; ++i) {
      double angle = 2 * M_PI * i / n;
      Eigen::AngleAxisd rotation(angle, Eigen::Vector3d::UnitZ());
      rotations.emplace_back(rotation); // Assuming Rotation takes (x, y, z)
    }

    std::map<std::pair<size_t, size_t>, Eigen::Quaterniond> relative_rotations;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                Eigen::Quaterniond relative_rotation = rotations[j] * rotations[i].inverse();
                relative_rotations[{i, j}] = relative_rotation;
            }
        }
    }

    // create connection Laplacia of relative_rotations
    // Eigen::MatrixXd L(3 * n , 3 * n);
    // L.setZeros();
    
    // for (auto& [edges_index, relative_rotation] : relative_rotations) {
    //     if (edges_index.first == edges_index.second) {
    //         L.block<3, 3>(edges_index.first * 3, edges_index.second * 3) = Eigen::Matrix3d::Identity();
    //     } else {
    //         L.block<3, 3>(edges_index.first * 3, edges_index.second * 3) = relative_rotation;
    //         L.block<3, 3>(edges_index.second* 3, edges_index.first* 3) = relative_rotation.inverse();
    //     }
    // }

}

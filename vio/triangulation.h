#ifndef VIO_TRIANGULATION_H_
#define VIO_TRIANGULATION_H_
#include <Eigen/Dense>

namespace vio {

// Triangulates a rectified stereo correspondence into the rectified-cam0
// frame (origin at cam0's optical center, axes = the shared rectified
// orientation). disparity = u_left - u_right, expected > 0 for points in
// front of both cameras. Returns false and leaves *point untouched if
// disparity <= min_disparity.
bool TriangulateRectified(double u_left, double v_left, double disparity, double fx, double fy,
                          double cx, double cy, double baseline_m, double min_disparity,
                          Eigen::Vector3d* point);

}  // namespace vio
#endif  // VIO_TRIANGULATION_H_

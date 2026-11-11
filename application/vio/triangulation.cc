#include "triangulation.h"

namespace vio {

bool TriangulateRectified(double u_left, double v_left, double disparity, double fx, double fy,
                          double cx, double cy, double baseline_m, double min_disparity,
                          Eigen::Vector3d* point) {
  if (disparity <= min_disparity) return false;

  const double z = fx * baseline_m / disparity;
  const double x = (u_left - cx) * z / fx;
  const double y = (v_left - cy) * z / fy;
  *point = Eigen::Vector3d(x, y, z);
  return true;
}

}  // namespace vio

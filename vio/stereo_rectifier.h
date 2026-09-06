#ifndef VIO_STEREO_RECTIFIER_H_
#define VIO_STEREO_RECTIFIER_H_
#include <Eigen/Dense>

#include "euroc_types.h"
#include "image_io.h"
#include "sophus/se3.hpp"

namespace vio {

// Forward radial-tangential distortion of a normalized (undistorted, z=1
// plane) point. Public and independently testable -- also used internally
// to build the rectification remap tables.
Eigen::Vector2d DistortRadTan(const Eigen::Vector2d& normalized_point, const RadTanDistortion& d);

struct RectifyMap {
  Eigen::MatrixXf map_u;  // rows x cols: source raw-image x per output pixel
  Eigen::MatrixXf map_v;  // rows x cols: source raw-image y per output pixel
};

struct StereoRectification {
  int rows = 0, cols = 0;
  double fx = 0, fy = 0, cx = 0, cy = 0;  // shared rectified intrinsics
  double baseline_m = 0;
  Sophus::SE3d T_body_rectcam0;  // fixed extrinsic: rectified-cam0 frame -> body
  RectifyMap map0, map1;

  // camera_index: 0 for cam0, 1 for cam1. Bilinear-samples raw via the
  // matching map; out-of-source-bounds output pixels are set to 0.
  GrayImage Rectify(const GrayImage& raw, int camera_index) const;
};

// output_rows/output_cols default to matching the input calibration's
// width/height (no valid-region cropping in v1 -- documented simplification).
StereoRectification ComputeStereoRectification(const CameraCalibration& cam0,
                                               const CameraCalibration& cam1, int output_rows,
                                               int output_cols);

}  // namespace vio
#endif  // VIO_STEREO_RECTIFIER_H_

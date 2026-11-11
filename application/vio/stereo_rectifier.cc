#include "stereo_rectifier.h"

#include <algorithm>
#include <cmath>

namespace vio {

Eigen::Vector2d DistortRadTan(const Eigen::Vector2d& normalized_point, const RadTanDistortion& d) {
  const double x = normalized_point.x();
  const double y = normalized_point.y();
  const double r2 = x * x + y * y;
  const double radial = 1.0 + d.k1 * r2 + d.k2 * r2 * r2;
  const double x_d = x * radial + 2.0 * d.p1 * x * y + d.p2 * (r2 + 2.0 * x * x);
  const double y_d = y * radial + d.p1 * (r2 + 2.0 * y * y) + 2.0 * d.p2 * x * y;
  return Eigen::Vector2d(x_d, y_d);
}

namespace {

RectifyMap BuildRemap(const Eigen::Matrix3d& R_rectify, const CameraCalibration& cam,
                      const StereoRectification& shared, int output_rows, int output_cols) {
  RectifyMap map;
  map.map_u = Eigen::MatrixXf(output_rows, output_cols);
  map.map_v = Eigen::MatrixXf(output_rows, output_cols);

  const Eigen::Matrix3d R_rectify_T = R_rectify.transpose();
  for (int v_r = 0; v_r < output_rows; ++v_r) {
    for (int u_r = 0; u_r < output_cols; ++u_r) {
      const double x_n = (u_r - shared.cx) / shared.fx;
      const double y_n = (v_r - shared.cy) / shared.fy;
      const Eigen::Vector3d ray_rect(x_n, y_n, 1.0);
      const Eigen::Vector3d ray_orig = R_rectify_T * ray_rect;
      const double x_p = ray_orig.x() / ray_orig.z();
      const double y_p = ray_orig.y() / ray_orig.z();
      const Eigen::Vector2d distorted = DistortRadTan(Eigen::Vector2d(x_p, y_p), cam.distortion);
      map.map_u(v_r, u_r) =
          static_cast<float>(cam.intrinsics.fu * distorted.x() + cam.intrinsics.cu);
      map.map_v(v_r, u_r) =
          static_cast<float>(cam.intrinsics.fv * distorted.y() + cam.intrinsics.cv);
    }
  }
  return map;
}

}  // namespace

GrayImage StereoRectification::Rectify(const GrayImage& raw, int camera_index) const {
  const RectifyMap& map = camera_index == 0 ? map0 : map1;

  GrayImage output;
  output.rows = rows;
  output.cols = cols;
  output.pixels.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);

  for (int v = 0; v < rows; ++v) {
    for (int u = 0; u < cols; ++u) {
      const double sample = raw.Bilinear(map.map_v(v, u), map.map_u(v, u));
      const double clamped = std::min(255.0, std::max(0.0, sample));
      output.pixels[static_cast<size_t>(v) * cols + u] =
          static_cast<uint8_t>(std::lround(clamped));
    }
  }
  return output;
}

StereoRectification ComputeStereoRectification(const CameraCalibration& cam0,
                                               const CameraCalibration& cam1, int output_rows,
                                               int output_cols) {
  const Eigen::Matrix3d R_BS0 = cam0.T_BS.rotationMatrix();
  const Eigen::Matrix3d R_BS1 = cam1.T_BS.rotationMatrix();
  const Eigen::Vector3d C0 = cam0.T_BS.translation();
  const Eigen::Vector3d C1 = cam1.T_BS.translation();

  const Eigen::Vector3d baseline_vec = C1 - C0;

  StereoRectification result;
  result.rows = output_rows;
  result.cols = output_cols;
  result.baseline_m = baseline_vec.norm();
  result.fx = 0.5 * (cam0.intrinsics.fu + cam1.intrinsics.fu);
  result.fy = 0.5 * (cam0.intrinsics.fv + cam1.intrinsics.fv);
  result.cx = 0.5 * (cam0.intrinsics.cu + cam1.intrinsics.cu);
  result.cy = 0.5 * (cam0.intrinsics.cv + cam1.intrinsics.cv);

  const Eigen::Vector3d e1 = baseline_vec.normalized();
  const Eigen::Vector3d e2 = (R_BS0.col(2).cross(e1)).normalized();
  const Eigen::Vector3d e3 = e1.cross(e2);
  Eigen::Matrix3d R_rect_body;
  R_rect_body.row(0) = e1.transpose();
  R_rect_body.row(1) = e2.transpose();
  R_rect_body.row(2) = e3.transpose();

  const Eigen::Matrix3d R_rectify0 = R_rect_body * R_BS0;
  const Eigen::Matrix3d R_rectify1 = R_rect_body * R_BS1;

  result.T_body_rectcam0 = Sophus::SE3d(Sophus::SO3d::fitToSO3(R_rect_body.transpose()), C0);

  result.map0 = BuildRemap(R_rectify0, cam0, result, output_rows, output_cols);
  result.map1 = BuildRemap(R_rectify1, cam1, result, output_rows, output_cols);

  return result;
}

}  // namespace vio

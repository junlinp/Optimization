#ifndef VIO_MONO_FEATURE_TRACKER_H_
#define VIO_MONO_FEATURE_TRACKER_H_
#include <vector>

#include <Eigen/Dense>

#include "euroc_types.h"
#include "harris_corners.h"
#include "image_io.h"
#include "patch_matcher.h"
#include "sophus/so3.hpp"

namespace vio {

// Inverse of DistortRadTan: distorted normalized point -> undistorted one,
// by fixed-point iteration (converges for EuRoC-strength distortion).
Eigen::Vector2d UndistortRadTan(const Eigen::Vector2d& distorted, const RadTanDistortion& d);

struct MonoTrackerOptions {
  HarrisOptions harris;
  PatchMatchOptions match = [] {
    PatchMatchOptions o;
    o.temporal_subpixel = true;
    return o;
  }();
  int max_features = 50;
  // Half-width of the 2D search window around the predicted location. With
  // a gyro-predicted rotation this can be much smaller than
  // PatchMatchOptions::temporal_search_radius, which is centred on the
  // feature's previous location.
  int search_radius = 12;
  // A forward match is kept only if matching it back into the previous
  // image lands within this many pixels of where it started.
  double forward_backward_threshold_px = 1.0;
  int border_px = 8;
};

struct TrackedFeature {
  int id = 0;
  double u = 0, v = 0;  // raw (distorted) cam pixel
  Eigen::Vector3d bearing = Eigen::Vector3d::UnitZ();  // unit vector, camera frame
};

// Monocular feature tracker with persistent ids, the front end EqVIO
// consumes (the paper uses GIFT; this is a hand-rolled stand-in built from
// the same Harris + SSD patch pieces as StereoVoFrontend). Tracking runs on
// the raw image; only the tracked points are undistorted, into bearings.
class MonoFeatureTracker {
 public:
  MonoFeatureTracker(const CameraCalibration& camera, const MonoTrackerOptions& options);

  // R_prev_curr: rotation of the current camera frame expressed in the
  // previous one (e.g. from the filter's IMU propagation), used to predict
  // where each feature moved. Pass identity when unknown.
  std::vector<TrackedFeature> Track(const GrayImage& image, const Sophus::SO3d& R_prev_curr);

  Eigen::Vector3d PixelToBearing(double u, double v) const;
  // Returns false if the point is behind the camera.
  bool BearingToPixel(const Eigen::Vector3d& bearing, double* u, double* v) const;

 private:
  bool InsideBorder(double u, double v) const;

  CameraCalibration camera_;
  MonoTrackerOptions options_;
  std::vector<TrackedFeature> features_;
  GrayImage prev_image_;
  bool have_prev_ = false;
  int next_id_ = 0;
};

}  // namespace vio
#endif  // VIO_MONO_FEATURE_TRACKER_H_

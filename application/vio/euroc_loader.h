#ifndef VIO_EUROC_LOADER_H_
#define VIO_EUROC_LOADER_H_
#include <string>
#include <vector>

#include "euroc_types.h"

namespace vio {

// "#timestamp [ns],filename" -> CameraFrameEntry rows, sorted by timestamp.
std::vector<CameraFrameEntry> LoadCameraIndex(const std::string& data_csv_path);

// "#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z" -> ImuSample rows, sorted.
std::vector<ImuSample> LoadImuData(const std::string& data_csv_path);

// state_groundtruth_estimate0/data.csv -> GroundTruthSample rows, sorted.
// Column order is p(3), q_w,q_x,q_y,q_z, v(3), b_w(3), b_a(3).
//
// Sophus::SO3d is built as Sophus::SO3d(Eigen::Quaterniond(qw, qx, qy, qz))
// -- Eigen::Quaterniond's constructor takes (w,x,y,z), matching the file's
// column order directly, but .coeffs() on the resulting object returns
// [x,y,z,w]. Do not pass the raw file order into .coeffs() or a raw
// Eigen::Vector4d constructor.
std::vector<GroundTruthSample> LoadGroundTruth(const std::string& data_csv_path);

// Reads one sensor.yaml via ParseYamlLite and extracts T_BS/intrinsics/
// distortion_coefficients/resolution/rate_hz. Throws std::runtime_error if a
// required key is missing or a list has the wrong length.
CameraCalibration LoadCameraSensorYaml(const std::string& sensor_yaml_path);
ImuCalibration LoadImuSensorYaml(const std::string& sensor_yaml_path);

// Convenience aggregate: expects mav0_dir/{cam0,cam1,imu0,
// state_groundtruth_estimate0} with the standard EuRoC file names.
EurocSequence LoadEurocSequence(const std::string& mav0_dir);

}  // namespace vio
#endif  // VIO_EUROC_LOADER_H_

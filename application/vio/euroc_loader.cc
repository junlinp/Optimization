#include "euroc_loader.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "yaml_lite.h"

namespace vio {
namespace {

std::string Trim(const std::string& s) {
  size_t begin = s.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return "";
  size_t end = s.find_last_not_of(" \t\r\n");
  return s.substr(begin, end - begin + 1);
}

std::vector<std::string> CommaTokens(const std::string& line) {
  std::vector<std::string> tokens;
  std::istringstream iss(line);
  std::string token;
  while (std::getline(iss, token, ',')) {
    tokens.push_back(Trim(token));
  }
  return tokens;
}

double ParseDoubleOrThrow(const std::string& token, const std::string& context) {
  if (token.empty()) {
    throw std::runtime_error("euroc_loader: empty numeric field in " + context);
  }
  char* end = nullptr;
  double value = std::strtod(token.c_str(), &end);
  if (end != token.c_str() + token.size()) {
    throw std::runtime_error("euroc_loader: malformed number '" + token + "' in " + context);
  }
  return value;
}

int64_t ParseInt64OrThrow(const std::string& token, const std::string& context) {
  if (token.empty()) {
    throw std::runtime_error("euroc_loader: empty integer field in " + context);
  }
  char* end = nullptr;
  long long value = std::strtoll(token.c_str(), &end, 10);
  if (end != token.c_str() + token.size()) {
    throw std::runtime_error("euroc_loader: malformed integer '" + token + "' in " + context);
  }
  return static_cast<int64_t>(value);
}

// Common driver for the two data.csv formats: skips blank/comment lines,
// hands each remaining line's comma tokens to `parse_row`, sorts the result
// by timestamp_ns. `parse_row` throws std::runtime_error (with the raw line
// folded into the message) on a malformed row.
template <class Row, class ParseFn>
std::vector<Row> ReadCsvRows(const std::string& path, ParseFn parse_row) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    throw std::runtime_error("euroc_loader: could not open file: " + path);
  }

  std::vector<Row> rows;
  std::string line;
  while (std::getline(ifs, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed[0] == '#') continue;
    rows.push_back(parse_row(CommaTokens(trimmed), path + ": " + trimmed));
  }
  std::sort(rows.begin(), rows.end(),
           [](const Row& a, const Row& b) { return a.timestamp_ns < b.timestamp_ns; });
  return rows;
}

const std::vector<double>& RequireList(const YamlLite& yaml, const std::string& key,
                                       size_t expected_size, const std::string& path) {
  auto it = yaml.number_lists.find(key);
  if (it == yaml.number_lists.end()) {
    throw std::runtime_error("euroc_loader: missing key '" + key + "' in " + path);
  }
  if (it->second.size() != expected_size) {
    throw std::runtime_error("euroc_loader: key '" + key + "' in " + path + " has " +
                             std::to_string(it->second.size()) + " values, expected " +
                             std::to_string(expected_size));
  }
  return it->second;
}

const std::string& RequireScalar(const YamlLite& yaml, const std::string& key,
                                 const std::string& path) {
  auto it = yaml.scalars.find(key);
  if (it == yaml.scalars.end()) {
    throw std::runtime_error("euroc_loader: missing key '" + key + "' in " + path);
  }
  return it->second;
}

Sophus::SE3d ParseTBS(const YamlLite& yaml, const std::string& path) {
  const std::vector<double>& data = RequireList(yaml, "T_BS.data", 16, path);
  Eigen::Matrix4d M;
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      M(row, col) = data[static_cast<size_t>(row) * 4 + col];
    }
  }
  return Sophus::SE3d(Sophus::SO3d::fitToSO3(M.block<3, 3>(0, 0)), M.block<3, 1>(0, 3));
}

}  // namespace

std::vector<CameraFrameEntry> LoadCameraIndex(const std::string& data_csv_path) {
  return ReadCsvRows<CameraFrameEntry>(
      data_csv_path, [&](const std::vector<std::string>& tokens, const std::string& context) {
        if (tokens.size() != 2) {
          throw std::runtime_error("euroc_loader: expected 2 fields in " + context);
        }
        CameraFrameEntry entry;
        entry.timestamp_ns = ParseInt64OrThrow(tokens[0], context);
        entry.filename = tokens[1];
        return entry;
      });
}

std::vector<ImuSample> LoadImuData(const std::string& data_csv_path) {
  return ReadCsvRows<ImuSample>(
      data_csv_path, [&](const std::vector<std::string>& tokens, const std::string& context) {
        if (tokens.size() != 7) {
          throw std::runtime_error("euroc_loader: expected 7 fields in " + context);
        }
        ImuSample sample;
        sample.timestamp_ns = ParseInt64OrThrow(tokens[0], context);
        sample.gyro = Eigen::Vector3d(ParseDoubleOrThrow(tokens[1], context),
                                      ParseDoubleOrThrow(tokens[2], context),
                                      ParseDoubleOrThrow(tokens[3], context));
        sample.accel = Eigen::Vector3d(ParseDoubleOrThrow(tokens[4], context),
                                       ParseDoubleOrThrow(tokens[5], context),
                                       ParseDoubleOrThrow(tokens[6], context));
        return sample;
      });
}

std::vector<GroundTruthSample> LoadGroundTruth(const std::string& data_csv_path) {
  return ReadCsvRows<GroundTruthSample>(
      data_csv_path, [&](const std::vector<std::string>& tokens, const std::string& context) {
        if (tokens.size() != 17) {
          throw std::runtime_error("euroc_loader: expected 17 fields in " + context);
        }
        GroundTruthSample sample;
        sample.timestamp_ns = ParseInt64OrThrow(tokens[0], context);
        sample.p_world = Eigen::Vector3d(ParseDoubleOrThrow(tokens[1], context),
                                         ParseDoubleOrThrow(tokens[2], context),
                                         ParseDoubleOrThrow(tokens[3], context));
        // File column order is q_w,q_x,q_y,q_z -- matches Eigen::Quaterniond's
        // (w,x,y,z) constructor argument order directly.
        const double qw = ParseDoubleOrThrow(tokens[4], context);
        const double qx = ParseDoubleOrThrow(tokens[5], context);
        const double qy = ParseDoubleOrThrow(tokens[6], context);
        const double qz = ParseDoubleOrThrow(tokens[7], context);
        sample.R_world_body = Sophus::SO3d(Eigen::Quaterniond(qw, qx, qy, qz));
        sample.v_world = Eigen::Vector3d(ParseDoubleOrThrow(tokens[8], context),
                                         ParseDoubleOrThrow(tokens[9], context),
                                         ParseDoubleOrThrow(tokens[10], context));
        sample.bias_gyro = Eigen::Vector3d(ParseDoubleOrThrow(tokens[11], context),
                                           ParseDoubleOrThrow(tokens[12], context),
                                           ParseDoubleOrThrow(tokens[13], context));
        sample.bias_accel = Eigen::Vector3d(ParseDoubleOrThrow(tokens[14], context),
                                            ParseDoubleOrThrow(tokens[15], context),
                                            ParseDoubleOrThrow(tokens[16], context));
        return sample;
      });
}

CameraCalibration LoadCameraSensorYaml(const std::string& sensor_yaml_path) {
  const YamlLite yaml = ParseYamlLite(sensor_yaml_path);

  CameraCalibration calib;
  calib.T_BS = ParseTBS(yaml, sensor_yaml_path);

  const std::vector<double>& intrinsics = RequireList(yaml, "intrinsics", 4, sensor_yaml_path);
  calib.intrinsics = {intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]};

  const std::vector<double>& distortion =
      RequireList(yaml, "distortion_coefficients", 4, sensor_yaml_path);
  calib.distortion = {distortion[0], distortion[1], distortion[2], distortion[3]};

  const std::vector<double>& resolution = RequireList(yaml, "resolution", 2, sensor_yaml_path);
  calib.width = static_cast<int>(resolution[0]);
  calib.height = static_cast<int>(resolution[1]);

  calib.rate_hz = ParseDoubleOrThrow(RequireScalar(yaml, "rate_hz", sensor_yaml_path),
                                     sensor_yaml_path);
  return calib;
}

ImuCalibration LoadImuSensorYaml(const std::string& sensor_yaml_path) {
  const YamlLite yaml = ParseYamlLite(sensor_yaml_path);

  ImuCalibration calib;
  calib.rate_hz =
      ParseDoubleOrThrow(RequireScalar(yaml, "rate_hz", sensor_yaml_path), sensor_yaml_path);
  calib.gyro_noise_density = ParseDoubleOrThrow(
      RequireScalar(yaml, "gyroscope_noise_density", sensor_yaml_path), sensor_yaml_path);
  calib.gyro_random_walk = ParseDoubleOrThrow(
      RequireScalar(yaml, "gyroscope_random_walk", sensor_yaml_path), sensor_yaml_path);
  calib.accel_noise_density = ParseDoubleOrThrow(
      RequireScalar(yaml, "accelerometer_noise_density", sensor_yaml_path), sensor_yaml_path);
  calib.accel_random_walk = ParseDoubleOrThrow(
      RequireScalar(yaml, "accelerometer_random_walk", sensor_yaml_path), sensor_yaml_path);
  return calib;
}

EurocSequence LoadEurocSequence(const std::string& mav0_dir) {
  EurocSequence seq;
  seq.mav0_dir = mav0_dir;
  seq.cam0 = LoadCameraSensorYaml(mav0_dir + "/cam0/sensor.yaml");
  seq.cam1 = LoadCameraSensorYaml(mav0_dir + "/cam1/sensor.yaml");
  seq.imu0 = LoadImuSensorYaml(mav0_dir + "/imu0/sensor.yaml");
  seq.imu_samples = LoadImuData(mav0_dir + "/imu0/data.csv");
  seq.cam0_frames = LoadCameraIndex(mav0_dir + "/cam0/data.csv");
  seq.cam1_frames = LoadCameraIndex(mav0_dir + "/cam1/data.csv");
  seq.ground_truth = LoadGroundTruth(mav0_dir + "/state_groundtruth_estimate0/data.csv");
  return seq;
}

}  // namespace vio

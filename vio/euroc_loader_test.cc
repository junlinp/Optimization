#include "euroc_loader.h"

#include <fstream>

#include "gtest/gtest.h"
#include "yaml_lite.h"

namespace vio {
namespace {

// Mirrors super_resolution.cc's FindLena candidate-path pattern: Bazel's test
// working directory/runfiles layout isn't guaranteed to be the repo root.
std::string FindTestdataMav0Dir() {
  const std::vector<std::string> candidates = {
      "vio/testdata/mav0",
      "testdata/mav0",
      std::string(std::getenv("TEST_SRCDIR") ? std::getenv("TEST_SRCDIR") : ".") +
          "/Optimization/vio/testdata/mav0",
      std::string(std::getenv("TEST_SRCDIR") ? std::getenv("TEST_SRCDIR") : ".") +
          "/_main/vio/testdata/mav0",
  };
  for (const auto& path : candidates) {
    std::ifstream probe(path + "/cam0/data.csv");
    if (probe.good()) return path;
  }
  return candidates.front();
}

}  // namespace

TEST(EurocLoader, LoadCameraSensorYamlParsesRealCam0Fixture) {
  const CameraCalibration calib =
      LoadCameraSensorYaml(FindTestdataMav0Dir() + "/cam0/sensor.yaml");

  const Eigen::Vector3d t = calib.T_BS.translation();
  EXPECT_NEAR(t.x(), -0.0216401454975, 1e-12);
  EXPECT_NEAR(t.y(), -0.064676986768, 1e-12);
  EXPECT_NEAR(t.z(), 0.00981073058949, 1e-12);

  const Eigen::Matrix3d R = calib.T_BS.rotationMatrix();
  EXPECT_NEAR(R(0, 0), 0.0148655429818, 1e-9);
  EXPECT_NEAR(R(0, 1), -0.999880929698, 1e-9);
  EXPECT_NEAR(R(2, 2), 0.999660727178, 1e-9);

  EXPECT_NEAR(calib.intrinsics.fu, 458.654, 1e-9);
  EXPECT_NEAR(calib.intrinsics.fv, 457.296, 1e-9);
  EXPECT_NEAR(calib.intrinsics.cu, 367.215, 1e-9);
  EXPECT_NEAR(calib.intrinsics.cv, 248.375, 1e-9);

  EXPECT_NEAR(calib.distortion.k1, -0.28340811, 1e-12);
  EXPECT_NEAR(calib.distortion.k2, 0.07395907, 1e-12);
  EXPECT_NEAR(calib.distortion.p1, 0.00019359, 1e-12);
  EXPECT_NEAR(calib.distortion.p2, 1.76187114e-05, 1e-16);

  EXPECT_EQ(calib.width, 752);
  EXPECT_EQ(calib.height, 480);
  EXPECT_NEAR(calib.rate_hz, 20.0, 1e-12);
}

TEST(EurocLoader, LoadImuSensorYamlParsesRealImu0Fixture) {
  const ImuCalibration calib = LoadImuSensorYaml(FindTestdataMav0Dir() + "/imu0/sensor.yaml");

  EXPECT_NEAR(calib.rate_hz, 200.0, 1e-12);
  EXPECT_NEAR(calib.gyro_noise_density, 1.6968e-04, 1e-18);
  EXPECT_NEAR(calib.gyro_random_walk, 1.9393e-05, 1e-19);
  EXPECT_NEAR(calib.accel_noise_density, 2.0000e-3, 1e-17);
  EXPECT_NEAR(calib.accel_random_walk, 3.0000e-3, 1e-17);
}

TEST(EurocLoader, LoadCameraIndexParsesRows) {
  const std::vector<CameraFrameEntry> frames =
      LoadCameraIndex(FindTestdataMav0Dir() + "/cam0/data.csv");

  ASSERT_EQ(frames.size(), 3u);
  EXPECT_EQ(frames[0].timestamp_ns, 1403715273262142976LL);
  EXPECT_EQ(frames[0].filename, "1403715273262142976.png");
  EXPECT_EQ(frames[2].timestamp_ns, 1403715273362142976LL);
}

TEST(EurocLoader, LoadImuDataParsesRows) {
  const std::vector<ImuSample> samples = LoadImuData(FindTestdataMav0Dir() + "/imu0/data.csv");

  ASSERT_EQ(samples.size(), 3u);
  EXPECT_EQ(samples[0].timestamp_ns, 1403715273262142976LL);
  EXPECT_NEAR(samples[0].gyro.x(), -0.0020943951023931952, 1e-15);
  EXPECT_NEAR(samples[0].gyro.z(), 0.07749261878854824, 1e-15);
  EXPECT_NEAR(samples[0].accel.x(), 9.0874956666666655, 1e-12);
  EXPECT_NEAR(samples[0].accel.z(), -3.6938381666666662, 1e-12);
}

TEST(EurocLoader, LoadGroundTruthParsesQuaternionOrderCorrectly) {
  const std::vector<GroundTruthSample> gt =
      LoadGroundTruth(FindTestdataMav0Dir() + "/state_groundtruth_estimate0/data.csv");

  ASSERT_EQ(gt.size(), 3u);
  const GroundTruthSample& s = gt[0];
  EXPECT_EQ(s.timestamp_ns, 1403715274302142976LL);
  EXPECT_NEAR(s.p_world.x(), 0.878612, 1e-9);
  EXPECT_NEAR(s.p_world.y(), 2.142470, 1e-9);
  EXPECT_NEAR(s.p_world.z(), 0.947262, 1e-9);

  // The file's column order is q_w,q_x,q_y,q_z = 0.060514,-0.828459,
  // -0.058956,-0.553641. Eigen's .coeffs() returns [x,y,z,w], so this is
  // the direct regression test for the constructor-argument-order gotcha
  // documented in euroc_loader.h. Tolerance is 1e-6, not tighter, because
  // the file's quaternion (only 6 decimal digits) has norm 1.0000002 --
  // Eigen::Quaterniond's constructor normalizes it, so .coeffs() differs
  // from the raw file values by that renormalization, not by a parsing bug.
  const Eigen::Vector4d coeffs = s.R_world_body.unit_quaternion().coeffs();
  EXPECT_NEAR(coeffs(0), -0.828459, 1e-6);  // x
  EXPECT_NEAR(coeffs(1), -0.058956, 1e-6);  // y
  EXPECT_NEAR(coeffs(2), -0.553641, 1e-6);  // z
  EXPECT_NEAR(coeffs(3), 0.060514, 1e-6);   // w

  EXPECT_NEAR(s.v_world.x(), 0.009474, 1e-9);
  EXPECT_NEAR(s.v_world.y(), -0.014009, 1e-9);
  EXPECT_NEAR(s.v_world.z(), -0.002145, 1e-9);

  // Field order after v is b_w(x,y,z) then b_a(x,y,z).
  EXPECT_NEAR(s.bias_gyro.x(), -0.002229, 1e-9);
  EXPECT_NEAR(s.bias_gyro.y(), 0.020700, 1e-9);
  EXPECT_NEAR(s.bias_gyro.z(), 0.076350, 1e-9);
  EXPECT_NEAR(s.bias_accel.x(), -0.012492, 1e-9);
  EXPECT_NEAR(s.bias_accel.y(), 0.547666, 1e-9);
  EXPECT_NEAR(s.bias_accel.z(), 0.069073, 1e-9);
}

TEST(EurocLoader, LoadEurocSequenceEndToEndOnFixtureDirectory) {
  const EurocSequence seq = LoadEurocSequence(FindTestdataMav0Dir());

  EXPECT_EQ(seq.imu_samples.size(), 3u);
  EXPECT_EQ(seq.cam0_frames.size(), 3u);
  EXPECT_EQ(seq.cam1_frames.size(), 3u);
  EXPECT_EQ(seq.ground_truth.size(), 3u);

  EXPECT_EQ(seq.cam0.width, 752);
  EXPECT_EQ(seq.cam1.width, 752);
  EXPECT_NEAR(seq.imu0.gyro_noise_density, 1.6968e-04, 1e-18);

  // cam0 and cam1 have distinct extrinsics/intrinsics -- confirms both were
  // parsed from their own files, not one overwriting the other.
  EXPECT_NE(seq.cam0.intrinsics.fu, seq.cam1.intrinsics.fu);
}

}  // namespace vio

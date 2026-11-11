#include "vio_simulator.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace vio {
namespace {

Eigen::Vector3d Gaussian(std::mt19937& rng, double stddev) {
  if (stddev <= 0.0) return Eigen::Vector3d::Zero();
  std::normal_distribution<double> n(0.0, stddev);
  return {n(rng), n(rng), n(rng)};
}

std::vector<Eigen::Vector3d> ScatterOnCube(double h, int per_face, std::mt19937& rng) {
  std::uniform_real_distribution<double> u(-h, h);
  std::vector<Eigen::Vector3d> points;
  for (int axis = 0; axis < 3; ++axis) {
    for (const double side : {-h, h}) {
      for (int k = 0; k < per_face; ++k) {
        Eigen::Vector3d p(u(rng), u(rng), u(rng));
        p(axis) = side;
        points.push_back(p);
      }
    }
  }
  return points;
}

}  // namespace

VioSimulation SimulateVio(const VioSimConfig& config) {
  if (config.duration_s <= 0 || config.imu_hz <= 0 || config.camera_hz <= 0) {
    throw std::invalid_argument("duration and rates must be positive");
  }
  const int imu_per_frame = static_cast<int>(std::lround(config.imu_hz / config.camera_hz));
  if (imu_per_frame < 1 || std::abs(imu_per_frame * config.camera_hz - config.imu_hz) > 1e-9) {
    throw std::invalid_argument("camera_hz must divide imu_hz");
  }

  std::mt19937 rng(config.seed);
  VioSimulation sim;
  sim.T_body_camera = config.T_body_camera;
  sim.landmarks_world = ScatterOnCube(config.box_half_extent_m, config.landmarks_per_face, rng);

  const double dt = 1.0 / config.imu_hz;
  const std::size_t count =
      static_cast<std::size_t>(std::floor(config.duration_s * config.imu_hz)) + 1;
  const double gyro_sigma = config.noise.gyro_noise_density / std::sqrt(dt);
  const double accel_sigma = config.noise.accel_noise_density / std::sqrt(dt);
  const double gyro_rw = config.noise.gyro_random_walk * std::sqrt(dt);
  const double accel_rw = config.noise.accel_random_walk * std::sqrt(dt);
  Eigen::Vector3d bias_gyro = config.initial_bias_gyro;
  Eigen::Vector3d bias_accel = config.initial_bias_accel;

  const double kPi = 3.14159265358979323846;
  const Eigen::Vector3d rot_rate(0.25, 0.3, 0.2);
  const Eigen::Vector3d pos_rate(0.1 * kPi, 0.2 * kPi, 0.15 * kPi);

  for (std::size_t k = 0; k < count; ++k) {
    const double t = static_cast<double>(k) * dt;
    ImuTrajectorySample s;
    s.t = t;

    // R = Exp(phi(t)); body rate = R^T R_dot = J_r(phi) phi_dot, J_r = J_l^T.
    const Eigen::Vector3d phi = (kPi / 4) * (rot_rate * t).array().cos().matrix();
    const Eigen::Vector3d phi_dot =
        -(kPi / 4) * (rot_rate.array() * (rot_rate * t).array().sin()).matrix();
    s.R_true = Sophus::SO3d::exp(phi);
    s.gyro_true = SO3LeftJacobian(phi).transpose() * phi_dot;

    const Eigen::Array3d c = (pos_rate * t).array().cos();
    const Eigen::Array3d sn = (pos_rate * t).array().sin();
    s.p_true = 0.5 * c.matrix();
    s.v_true = -0.5 * (pos_rate.array() * sn).matrix();
    const Eigen::Vector3d a_world = -0.5 * (pos_rate.array().square() * c).matrix();
    s.accel_true = s.R_true.inverse() * (a_world - config.gravity_world);

    s.gyro_bias_true = bias_gyro;
    s.accel_bias_true = bias_accel;
    s.gyro_meas = s.gyro_true + bias_gyro;
    s.accel_meas = s.accel_true + bias_accel;
    if (config.add_imu_noise) {
      s.gyro_meas += Gaussian(rng, gyro_sigma);
      s.accel_meas += Gaussian(rng, accel_sigma);
    }
    sim.imu.push_back(s);
    if (config.add_bias_random_walk) {
      bias_gyro += Gaussian(rng, gyro_rw);
      bias_accel += Gaussian(rng, accel_rw);
    }
  }

  // Camera frames. Features already being tracked are kept while visible
  // (as a KLT tracker would), and new visible landmarks top the set up to
  // max_tracked_features, in a shuffled order.
  std::normal_distribution<double> pixel_noise(0.0, config.pixel_sigma);
  const double cx = 0.5 * config.width_px;
  const double cy = 0.5 * config.height_px;
  std::unordered_set<int> tracked;
  std::vector<int> order(sim.landmarks_world.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);

  for (std::size_t k = 0; k < count; k += imu_per_frame) {
    const ImuTrajectorySample& s = sim.imu[k];
    const Sophus::SE3d T_camera_world =
        (Sophus::SE3d(s.R_true, s.p_true) * config.T_body_camera).inverse();

    CameraFrame frame;
    frame.t = s.t;
    frame.imu_index = k;
    auto project = [&](int id, BearingMeasurement* out) {
      const Eigen::Vector3d q = T_camera_world * sim.landmarks_world[id];
      if (q.z() < config.min_depth_m) return false;
      double u = config.focal_px * q.x() / q.z() + cx;
      double v = config.focal_px * q.y() / q.z() + cy;
      if (u < 0 || v < 0 || u >= config.width_px || v >= config.height_px) return false;
      if (config.pixel_sigma > 0) {
        u += pixel_noise(rng);
        v += pixel_noise(rng);
      }
      out->id = id;
      out->bearing = Eigen::Vector3d((u - cx) / config.focal_px, (v - cy) / config.focal_px, 1.0)
                         .normalized();
      return true;
    };

    std::unordered_set<int> still_tracked;
    for (const int id : order) {
      if (!tracked.count(id)) continue;
      BearingMeasurement m;
      if (project(id, &m)) {
        frame.bearings.push_back(m);
        still_tracked.insert(id);
      }
    }
    std::shuffle(order.begin(), order.end(), rng);
    for (const int id : order) {
      if (static_cast<int>(frame.bearings.size()) >= config.max_tracked_features) break;
      if (still_tracked.count(id)) continue;
      BearingMeasurement m;
      if (project(id, &m)) {
        frame.bearings.push_back(m);
        still_tracked.insert(id);
      }
    }
    tracked = std::move(still_tracked);
    sim.frames.push_back(std::move(frame));
  }
  return sim;
}

EqvioState TruthState(const VioSimulation& sim, std::size_t k) {
  const ImuTrajectorySample& s = sim.imu[k];
  EqvioState xi;
  xi.R = s.R_true;
  xi.p = s.p_true;
  xi.v = s.v_true;
  xi.bias_gyro = s.gyro_bias_true;
  xi.bias_accel = s.accel_bias_true;
  xi.T_body_camera = sim.T_body_camera;
  const Sophus::SE3d T_camera_world = xi.T_world_camera().inverse();
  for (std::size_t i = 0; i < sim.landmarks_world.size(); ++i) {
    xi.landmark_ids.push_back(static_cast<int>(i));
    xi.landmarks_camera.push_back(T_camera_world * sim.landmarks_world[i]);
  }
  return xi;
}

}  // namespace vio

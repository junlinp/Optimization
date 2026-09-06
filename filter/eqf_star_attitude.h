#ifndef EQF_STAR_ATTITUDE_H
#define EQF_STAR_ATTITUDE_H

// EqF* (equivariant output linearisation) from:
//   van Goor, Hamel, Mahony, "Equivariant Filter (EqF)",
//   IEEE TAC 2022, https://arxiv.org/abs/2010.14666
//   Lemma 5.3, eq. (34)-(35); algorithm in Section 6.
//
// CDC 2020 preprint (arXiv:2107.05193) covers standard EqF only; EqF* appears
// in the journal version above.

#include <Eigen/Dense>
#include "eqf_attitude.h"
#include "sophus/so3.hpp"

namespace eqf_star_attitude {

using EqfStarAttitudeState = eqf_attitude::EqfAttitudeState;
using EqfStarImuNoiseParams = eqf_attitude::EqfImuNoiseParams;
using VectorMeasurement = eqf_attitude::VectorMeasurement;

class EqfStarAttitude {
 public:
  EqfStarAttitude(const EqfStarAttitudeState& initial_state,
                  const Eigen::Matrix<double, 3, 3>& initial_covariance,
                  const EqfStarImuNoiseParams& noise_params,
                  const Eigen::Vector3d& gravity_world_reference,
                  const Eigen::Vector3d& magnetometer_world_reference);
  ~EqfStarAttitude() = default;

  void Predict(const Eigen::Vector3d& gyro_meas, double dt);
  void UpdateVector(const VectorMeasurement& gravity,
                    const VectorMeasurement& magnetometer,
                    double dt);

  const EqfStarAttitudeState& state() const { return state_; }

  // EqF* output block for full SO(3) vector measurements (midpoint form).
  static Eigen::Matrix3d EquivariantOutputBlock(const Eigen::Vector3d& y_meas,
                                                const Eigen::Vector3d& y_pred,
                                                const Eigen::Matrix3d& R_hat);

 private:
  EqfStarAttitudeState state_;
  Eigen::Matrix<double, 3, 3> sigma_;
  Eigen::Matrix<double, 3, 3> P_;
  Eigen::Matrix<double, 6, 6> Q_;
  EqfStarImuNoiseParams noise_;
  Eigen::Vector3d dm_;
  Eigen::Vector3d dg_;
};

}  // namespace eqf_star_attitude

#endif  // EQF_STAR_ATTITUDE_H

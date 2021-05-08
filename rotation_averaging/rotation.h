#ifndef ROTATION_AVERAGING_ROTATION_H_
#define ROTATION_AVERAGING_ROTATION_H_

template<class T>
void quaternion_to_angle_axis(const T*quaternion, T* angle_aixs) {
    T theta_div_2 = std::acos(quaternion);
    T factor = 2 * theta_div_2 / std::sin(theta_div_2);
    angle_aixs[0] = quaternion[1] * factor;
    angle_aixs[1] = quaternion[2] * factor;
    angle_aixs[2] = quaternion[3] * factor;
}

template<class T>
void angle_axis_to_quaternion(const T* angle_axis, T* quaternion) {
  T norm = angle_axis[0] * angle_axis[0] + angle_axis[1] * angle_axis[1] +
           angle_axis[2] * angle_axis[2];
  norm = std::sqrt(norm);

  quaternion[0] = std::cos(norm / 2);
  T sine_norm_2 = std::sin(norm / 2);
  quaternion[1] = sine_norm_2 * angle_axis[0] / norm;
  quaternion[2] = sine_norm_2 * angle_axis[1] / norm;
  quaternion[3] = sine_norm_2 * angle_axis[2] / norm;
}

#endif  //ROTATION_AVERAGING_ROTATION_H_
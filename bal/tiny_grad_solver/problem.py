from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
import numpy as np


def SkewMatrix(vector3: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0, -vector3[2], vector3[1]],
            [vector3[2], 0, -vector3[0]],
            [-vector3[1], vector3[0], 0],
        ]
    )


def AngleAxisToRotationMatrix(angle_axis: np.ndarray) -> np.array:
    theta = np.linalg.norm(angle_axis)
    axis = np.divide(angle_axis, theta)
    skew_matrix = SkewMatrix(axis)
    return np.reshape(
        np.eye(3)
        + np.sin(theta) * SkewMatrix(axis)
        + (1 - np.cos(theta)) * (skew_matrix @ skew_matrix),
        (9),
    )


class Observation:
    def __init__(self, camera_id: int, point_id: int, x: float, y: float):
        self.camera_id = camera_id
        self.point_id = point_id
        self.u = x
        self.v = y
        self.l2_norm_uv = self.u * self.u + self.v * self.v

    def forward(self, camera_parameters: Tensor, point_parameters: Tensor) -> Tensor:
        camera_parameter = camera_parameters[
            self.camera_id * 15 : (self.camera_id + 1) * 15
        ]
        point_parameter = point_parameters[self.point_id * 3 : (self.point_id + 1) * 3]

        rotation_matrix = camera_parameter[0:9].reshape((3, 3))
        #print(f"rotation_matrix {rotation_matrix.numpy()}")
        #print(f"point {point_parameter.numpy()}")
        translation = camera_parameter[9:12]
        f = camera_parameter[12]
        k1 = camera_parameter[13]
        k2 = camera_parameter[14]
        rotated_point = rotation_matrix.matmul(point_parameter) + translation
        #print(f"rotated_point {rotated_point.numpy()}")
        #print(f"f :{f.numpy()}, k1 : {k1.numpy()}, k2 : {k2.numpy()}")
        t = f.add(f.mul(k1).mul(self.l2_norm_uv)).add(
            f.mul(k2).mul(self.l2_norm_uv * self.l2_norm_uv)
        )
        #print(f"t {t.numpy()}")
        ray_dot_rotated_point = (
            self.u * rotated_point[0] + self.v * rotated_point[1] + t * rotated_point[2]
        )
        length = ray_dot_rotated_point / rotated_point.dot(rotated_point)

        length_rotated_point = length * rotated_point

        return 0.5 * (
            (self.u - length_rotated_point[0]) * (self.u - length_rotated_point[0])
            + (self.v - length_rotated_point[1]) * (self.v - length_rotated_point[1])
            + (t - length_rotated_point[2]) * (t - length_rotated_point[2])
        )


class Problem:
    def __init__(self, data_path: str):
        self.camera_parameters = []
        self.point_parameters = []
        self.observations = []

        temp_camera_parameters = []
        temp_point_parameters = []
        with open(data_path, "r") as f:
            all_lines = f.readlines()
            [num_cameras, num_points, num_observations] = [
                int(n) for n in all_lines[0].split()
            ]
            print(
                f"{len(all_lines)} -> camera {num_cameras * 9}, point {num_points * 3}, observation {num_observations}"
            )
            assert (
                len(all_lines)
                == num_cameras * 9 + num_points * 3 + num_observations + 1
            )
            print(
                f"Camera {int(num_cameras)}, points {int(num_points)}, num_observations {(num_observations)}"
            )

            for line in all_lines[1 : 1 + num_observations]:
                [camera_id, point_id, x, y] = line.split()
                self.observations.append(
                    Observation(int(camera_id), int(point_id), float(x), float(y))
                )
            print("Process Observation Done")
            camera_lines = all_lines[1 + num_observations :]
            for i in range(int(num_cameras)):
                camera_parameters = [float(camera_lines[i * 9 + j]) for j in range(9)]
                rotation_matrix_flat = AngleAxisToRotationMatrix(
                    np.array(camera_parameters[0:3])
                )
                assert len(list(rotation_matrix_flat)) == 9
                assert len(camera_parameters[3:]) == 6
                temp_camera_parameters = (
                    temp_camera_parameters
                    + list(rotation_matrix_flat)
                    + camera_parameters[3:]
                )

            print("Process Camera Done")

            point_lines = all_lines[1 + num_observations + num_cameras * 9 :]
            temp_point_parameters = [float(line) for line in point_lines]
            print(f"{len(temp_point_parameters)} should be {num_points * 3}")
            print("Process Point Done")
            assert len(temp_camera_parameters) == int(num_cameras) * 15
            self.camera_parameters = Tensor(temp_camera_parameters, requires_grad=True)
            assert len(temp_point_parameters) == int(num_points) * 3
            self.point_parameters = Tensor(temp_point_parameters, requires_grad=True)
        self.optim = optim.SGD(
            [self.camera_parameters, self.point_parameters], lr=1e-30
        )

    def Step(self):
        loss = Tensor(0)
        for ob in self.observations:
            loss = loss.add(ob.forward(self.camera_parameters, self.point_parameters))
        print(f"loss : {loss.numpy()}")
        #self.optim.zero_grad()
        loss.backward()
        #print(self.camera_parameters.grad.numpy())
        #self.optim.step()

        #loss = Tensor(0)
        #for ob in self.observations[:128]:
        #    loss = loss.add(ob.forward(self.camera_parameters, self.point_parameters))
        #print(loss.numpy())

p = Problem("./problem-49-7776-pre.txt")
for i in range(8):
    p.Step()


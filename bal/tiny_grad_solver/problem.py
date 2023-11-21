from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
class Observation():
    def __init__(self, camera_id:int, point_id:int, x:float, y:float):
        self.camera_id = camera_id
        self.point_id = point_id
        self.obs = [x, y]

    def forward(self, camera_parameters, point_parameters):
        camera_parameter = camera_parameters.slice(self.camera_id * 9, (self.camera_id + 1) * 9)
        point_parameter = point_parameter.slice(self.point_id * 3, (self.point_id + 1) * 3)

        


     

class Problem:
    def __init__(self, data_path: str):

        self.camera_parameters = []
        self.point_parameters = []
        self.observations = []

        temp_camera_parameters = []
        temp_point_parameters = []
        with open(data_path, "r") as f:
            [num_cameras, num_points, num_observations] = f.readline().split()
            print(f"Camera {num_cameras}, points {num_points}, num_observations {num_observations}")

            for i in range(int(num_observations)):
                [camera_id, point_id, x, y] = f.readline().split()
                self.observations.append(Observation(camera_id, point_id, x, y))

            for i in range(int(num_cameras)):
                camera_parameters = f.readline().split()
                temp_camera_parameters = temp_camera_parameters + camera_parameters

            for i in range(int(num_points)):
                point_parameters = f.readline().split()
                temp_point_parameters = temp_point_parameters + point_parameters

            self.camera_parameters = Tensor(temp_camera_parameters, requires_grad=True)
            self.point_parameters = Tensor(temp_point_parameters, requires_grad=True)
        self.optim = optim.SGD([self.camera_parameters, self.point_parameters], lr=0.001)
    def Step(self):
        pass

p = Problem("/Users/panjunlin/repo/Optimization/problem-1723-156502-pre.txt")
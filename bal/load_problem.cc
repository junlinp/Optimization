#include "load_problem.h"
#include <fstream>
#include <iostream>

Problem LoadProblem(const std::string& path) {
    size_t num_cameras, num_points, num_observations;

    std::ifstream ifs(path);
    if (!ifs) {
        std::cout << "File " << path << " Can't Read." << std::endl;
        return Problem();
    }
    Problem problem;
    ifs >> num_cameras >> num_points >> num_observations;
    for(size_t i = 0; i < num_observations; ++i) {
        size_t camera_id, point_id;
        double x, y;
        ifs >> camera_id >> point_id >> x >> y;
        problem.observations_[IndexPair(camera_id, point_id)] = Observation(x, y);
    }

    for (size_t i = 0; i < num_cameras; i++) {
        double* data = problem.cameras_[i].params;
        for(int j = 0; j < 9; j++) {
            ifs >> data[j];
        }
    }

    for (size_t i = 0; i < num_points; i++) {
        double x, y, z;
        ifs >> x >> y >> z;
        problem.points_[i] = Landmark(x, y, z);
    }
    ifs.close();
    return problem;
}
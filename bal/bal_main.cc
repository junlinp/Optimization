#include <string>
#include "load_problem.h"
#include "iostream"
#include "evaluate.h"
#include "../JET.h"

#include "ceres/rotation.h"
#include "ceres/jet.h"

int main() {
    //const std::string path = "/Users/GEEK/Downloads/problem-1723-156502-pre.txt";
    const std::string path = "/Users/GEEK/Downloads/problem-49-7776-pre.txt";

    Problem problem = LoadProblem(path);
    std::cout << "Cameras : " << problem.cameras_.size() << std::endl;
    std::cout << "Points : " << problem.points_.size()  << std::endl;
    std::cout << "Observation : " << problem.observations_.size() << std::endl;
    Eigen::VectorXd e;
    Eigen::MatrixXd j;
    Evaluate(problem, e, j);
    return 0;
}
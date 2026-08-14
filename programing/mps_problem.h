#ifndef PROGRAMING_MPS_PROBLEM_H_
#define PROGRAMING_MPS_PROBLEM_H_

#include <Eigen/Dense>
#include <map>
#include <string>

struct MPSProblem {
    std::string name;
    std::map<std::string, char> row_types;
    std::map<std::string, int64_t> row_index;
    std::map<std::string, int64_t> col_index;

    std::map<int64_t, std::map<int64_t, double>> coefficients;

    std::map<int64_t, double> rhs;
    std::string objective_row_name;
    std::map<int64_t, double> objective_row_coefficients;
    std::map<int64_t, double> lower_bounds, upper_bounds;
};

MPSProblem read_mps(const std::string& filename);
void BuildDenseLp(const MPSProblem& prob, Eigen::MatrixXd* A, Eigen::VectorXd* b,
                  Eigen::VectorXd* c);

#endif  // PROGRAMING_MPS_PROBLEM_H_

#include <string>
#include <vector>
#include <map>

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
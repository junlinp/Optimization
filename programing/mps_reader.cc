#include "mps_problem.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

MPSProblem read_mps(const std::string& filename) {
    MPSProblem prob;
    std::string section;
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open file");
    std::string line;

    auto ensure_col = [&](const std::string& col) {
        auto it = prob.col_index.find(col);
        if (it == prob.col_index.end()) {
            int64_t index = static_cast<int64_t>(prob.col_index.size());
            it = prob.col_index.emplace(col, index).first;
        }
        return it->second;
    };

    auto add_entry = [&](const std::string& col, const std::string& row, double val) {
        int64_t j = ensure_col(col);
        if (row == prob.objective_row_name) {
            prob.objective_row_coefficients[j] = val;
            return;
        }
        auto row_it = prob.row_index.find(row);
        if (row_it != prob.row_index.end()) {
            prob.coefficients[row_it->second][j] = val;
        }
    };

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '*') continue;
        if (line.find("NAME") == 0) {
            section = "NAME";
            std::istringstream iss(line.substr(4));
            iss >> prob.name;
            continue;
        }
        if (line.find("ROWS") == 0) { section = "ROWS"; continue; }
        if (line.find("COLUMNS") == 0) { section = "COLUMNS"; continue; }
        if (line.find("RHS") == 0) { section = "RHS"; continue; }
        if (line.find("BOUNDS") == 0) { section = "BOUNDS"; continue; }
        if (line.find("ENDATA") == 0) break;
        std::istringstream iss(line);
        if (section == "ROWS") {
            char type; std::string row;
            iss >> type >> row;

            if (type == 'N') {
                prob.objective_row_name = row;
            } else if (type == 'E') {
                prob.row_types[row] = type;
                prob.row_index[row] = static_cast<int64_t>(prob.row_index.size());
            }
        } else if (section == "COLUMNS") {
            std::string col, row1, row2; double val1 = 0, val2 = 0;
            iss >> col >> row1 >> val1;
            if (row1 == "'MARKER'" || row1 == "MARKER") {
                continue;
            }
            add_entry(col, row1, val1);
            if (iss >> row2 >> val2) {
                add_entry(col, row2, val2);
            }
        } else if (section == "RHS") {
            std::string rhs_name, row; double val;
            iss >> rhs_name >> row >> val;
            if (prob.row_index.find(row) != prob.row_index.end()) {
                prob.rhs[prob.row_index[row]] = val;
            }
            if (iss >> row >> val) {
                if (prob.row_index.find(row) != prob.row_index.end()) {
                    prob.rhs[prob.row_index[row]] = val;
                }
            }
        } else if (section == "BOUNDS") {
            std::string bound_type, bound_name, var; double val = 0;
            iss >> bound_type >> bound_name >> var;
            if (prob.col_index.find(var) != prob.col_index.end()) {
                if (bound_type == "LO") { iss >> val; prob.lower_bounds[prob.col_index[var]] = val; }
                else if (bound_type == "UP") { iss >> val; prob.upper_bounds[prob.col_index[var]] = val; }
                else if (bound_type == "FX") { iss >> val; prob.lower_bounds[prob.col_index[var]] = val; prob.upper_bounds[prob.col_index[var]] = val; }
                else if (bound_type == "FR") { prob.lower_bounds[prob.col_index[var]] = -1e20; prob.upper_bounds[prob.col_index[var]] = 1e20; }
            }
        }
    }
    return prob;
}

void BuildDenseLp(const MPSProblem& prob, Eigen::MatrixXd* A, Eigen::VectorXd* b,
                  Eigen::VectorXd* c) {
    const Eigen::Index m = static_cast<Eigen::Index>(prob.row_index.size());
    const Eigen::Index n = static_cast<Eigen::Index>(prob.col_index.size());
    A->setZero(m, n);
    b->setZero(m);
    c->setZero(n);
    for (const auto& row_entry : prob.coefficients) {
        for (const auto& col_entry : row_entry.second) {
            (*A)(row_entry.first, col_entry.first) = col_entry.second;
        }
    }
    for (const auto& rhs_entry : prob.rhs) {
        (*b)(rhs_entry.first) = rhs_entry.second;
    }
    for (const auto& obj_entry : prob.objective_row_coefficients) {
        (*c)(obj_entry.first) = obj_entry.second;
    }
}

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "mps_problem.h"

MPSProblem read_mps(const std::string& filename) {
    MPSProblem prob;
    std::string section;
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open file");
    std::string line;
    while (std::getline(file, line)) {
        std::cout << line << std::endl;
        if (line.empty() || line[0] == '*') continue;
        if (line.find("NAME") == 0) { section = "NAME"; continue; }
        if (line.find("ROWS") == 0) { section = "ROWS"; continue; }
        if (line.find("COLUMNS") == 0) { section = "COLUMNS"; continue; }
        if (line.find("RHS") == 0) { section = "RHS"; continue; }
        if (line.find("BOUNDS") == 0) { section = "BOUNDS"; continue; }
        if (line.find("ENDATA") == 0) break;
        std::istringstream iss(line);
        if (section == "NAME") {
            iss >> prob.name;
            section.clear();
        } else if (section == "ROWS") {
            char type; std::string row;
            iss >> type >> row;

            if (type == 'N') {
                prob.objective_row_name = row;
            } else {
                if (type == 'E') {
                    prob.row_types[row] = type;
                    prob.row_index[row] = prob.row_index.size();
                }
            }
        } else if (section == "COLUMNS") {
            std::string col, row1, row2; double val1 = 0, val2 = 0;
            iss >> col >> row1 >> val1;
            if (row1 == prob.objective_row_name) {
                prob.objective_row_coefficients[prob.col_index[col]] = val1;
            }
            if (prob.col_index.find(col) == prob.col_index.end()) {
                prob.col_index[col] = prob.col_index.size();
            }
            if (prob.row_index.find(row1) != prob.row_index.end()) {
                prob.coefficients[prob.row_index[row1]][prob.col_index[col]] = val1;
            }
            if (iss >> row2 >> val2) {
                if (prob.row_index.find(row2) != prob.row_index.end()) {
                    prob.coefficients[prob.row_index[row2]][prob.col_index[col]] = val2;
                }
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

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <file.mps>\n";
//         return 1;
//     }
//     try {
//         MPSProblem prob = read_mps(argv[1]);
//         std::cout << "MPS file loaded: " << prob.name << "\n";
//         std::cout << "Rows: " << prob.row_names.size() << ", Cols: " << prob.col_names.size() << "\n";
//         std::cout << "Objective row: " << prob.objective_row << "\n";
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << "\n";
//         return 2;
//     }
//     return 0;
// }

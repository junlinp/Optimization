#include "mps_problem.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Sparse>

namespace {

constexpr double kBoundInf = 1e20;
constexpr double kInfThresh = 1e19;

std::string Trim(std::string s) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

std::string FixedField(const std::string& line, int col1, int col2) {
    if (static_cast<int>(line.size()) < col1) {
        return "";
    }
    const int begin = col1 - 1;
    const int len = std::min(col2, static_cast<int>(line.size())) - begin;
    if (len <= 0) {
        return "";
    }
    return Trim(line.substr(begin, static_cast<size_t>(len)));
}

std::vector<std::string> Tokens(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream iss(line);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

bool ParseDouble(const std::string& s, double* value) {
    if (s.empty()) {
        return false;
    }
    char* end = nullptr;
    *value = std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0';
}

bool IsMarker(const std::string& s) {
    return s == "'MARKER'" || s == "MARKER";
}

bool IsNegInf(double v) { return v <= -kInfThresh; }
bool IsPosInf(double v) { return v >= kInfThresh; }

void ConstraintBounds(char type, double rhs, bool has_range, double range,
                      double* lb, double* ub) {
    *lb = -kBoundInf;
    *ub = kBoundInf;
    if (type == 'E') {
        *lb = rhs;
        *ub = rhs;
        if (has_range) {
            if (range >= 0.0) {
                *ub = rhs + range;
            } else {
                *lb = rhs + range;
            }
        }
    } else if (type == 'L') {
        *ub = rhs;
        if (has_range) {
            *lb = rhs - std::abs(range);
        }
    } else if (type == 'G') {
        *lb = rhs;
        if (has_range) {
            *ub = rhs + std::abs(range);
        }
    }
}

struct VarMap {
    int pos = -1;
    int neg = -1;
    double shift = 0.0;
    double coeff_sign = 1.0;
};

}  // namespace

MPSProblem read_mps(const std::string& filename) {
    MPSProblem prob;
    std::string section;
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file");
    }
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
        if (IsMarker(row)) {
            return;
        }
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

    auto add_row_value = [&](std::map<int64_t, double>* dest, const std::string& row,
                             double val) {
        auto row_it = prob.row_index.find(row);
        if (row_it != prob.row_index.end()) {
            (*dest)[row_it->second] = val;
        }
    };

    auto parse_name_pairs = [&](const std::vector<std::string>& tok, size_t start,
                                std::map<int64_t, double>* dest) {
        for (size_t i = start; i + 1 < tok.size(); i += 2) {
            double val = 0.0;
            if (!ParseDouble(tok[i + 1], &val)) {
                return false;
            }
            add_row_value(dest, tok[i], val);
        }
        return tok.size() > start;
    };

    auto parse_fixed_pairs = [&](const std::string& line, std::map<int64_t, double>* dest) {
        const std::string r1 = FixedField(line, 15, 22);
        const std::string v1s = FixedField(line, 25, 36);
        double v1 = 0.0;
        if (!r1.empty() && ParseDouble(v1s, &v1)) {
            add_row_value(dest, r1, v1);
        }
        const std::string r2 = FixedField(line, 40, 47);
        const std::string v2s = FixedField(line, 50, 61);
        double v2 = 0.0;
        if (!r2.empty() && ParseDouble(v2s, &v2)) {
            add_row_value(dest, r2, v2);
        }
    };

    while (std::getline(file, line)) {
        auto dollar = line.find('$');
        if (dollar != std::string::npos) {
            line = line.substr(0, dollar);
        }
        if (line.empty() || line[0] == '*') {
            continue;
        }
        const std::string trimmed = Trim(line);
        if (trimmed.empty() || trimmed[0] == '*') {
            continue;
        }

        const bool header = !std::isspace(static_cast<unsigned char>(line[0]));
        if (header) {
            if (trimmed.compare(0, 4, "NAME") == 0) {
                section = "NAME";
                std::istringstream iss(trimmed.substr(4));
                iss >> prob.name;
                continue;
            }
            if (trimmed.compare(0, 4, "ROWS") == 0) {
                section = "ROWS";
                continue;
            }
            if (trimmed.compare(0, 7, "COLUMNS") == 0) {
                section = "COLUMNS";
                continue;
            }
            if (trimmed.compare(0, 3, "RHS") == 0) {
                section = "RHS";
                continue;
            }
            if (trimmed.compare(0, 6, "RANGES") == 0) {
                section = "RANGES";
                continue;
            }
            if (trimmed.compare(0, 6, "BOUNDS") == 0) {
                section = "BOUNDS";
                continue;
            }
            if (trimmed.compare(0, 6, "ENDATA") == 0) {
                break;
            }
            continue;
        }

        if (section == "ROWS") {
            std::istringstream iss(line);
            char type = 0;
            std::string row;
            iss >> type >> row;
            if (row.empty()) {
                continue;
            }
            type = static_cast<char>(std::toupper(static_cast<unsigned char>(type)));
            if (type == 'N') {
                if (prob.objective_row_name.empty()) {
                    prob.objective_row_name = row;
                }
            } else if (type == 'E' || type == 'L' || type == 'G') {
                if (prob.row_index.find(row) == prob.row_index.end()) {
                    prob.row_types[row] = type;
                    prob.row_index[row] = static_cast<int64_t>(prob.row_index.size());
                }
            }
        } else if (section == "COLUMNS") {
            const auto tok = Tokens(line);
            bool parsed = false;
            if (tok.size() >= 3 && !IsMarker(tok[1])) {
                parsed = true;
                const std::string& col = tok[0];
                for (size_t i = 1; i + 1 < tok.size(); i += 2) {
                    double val = 0.0;
                    if (!ParseDouble(tok[i + 1], &val)) {
                        parsed = false;
                        break;
                    }
                    add_entry(col, tok[i], val);
                }
            } else if (tok.size() >= 2 && IsMarker(tok[1])) {
                parsed = true;
            }
            if (!parsed) {
                const std::string col = FixedField(line, 5, 12);
                const std::string r1 = FixedField(line, 15, 22);
                if (IsMarker(r1)) {
                    continue;
                }
                const std::string v1s = FixedField(line, 25, 36);
                double v1 = 0.0;
                if (!col.empty() && !r1.empty() && ParseDouble(v1s, &v1)) {
                    add_entry(col, r1, v1);
                }
                const std::string r2 = FixedField(line, 40, 47);
                const std::string v2s = FixedField(line, 50, 61);
                double v2 = 0.0;
                if (!col.empty() && !r2.empty() && ParseDouble(v2s, &v2)) {
                    add_entry(col, r2, v2);
                }
            }
        } else if (section == "RHS") {
            const auto tok = Tokens(line);
            if (tok.size() >= 3 && !parse_name_pairs(tok, 1, &prob.rhs)) {
                parse_fixed_pairs(line, &prob.rhs);
            } else if (tok.size() < 3) {
                parse_fixed_pairs(line, &prob.rhs);
            }
        } else if (section == "RANGES") {
            const auto tok = Tokens(line);
            if (tok.size() >= 3 && !parse_name_pairs(tok, 1, &prob.ranges)) {
                parse_fixed_pairs(line, &prob.ranges);
            } else if (tok.size() < 3) {
                parse_fixed_pairs(line, &prob.ranges);
            }
        } else if (section == "BOUNDS") {
            auto tok = Tokens(line);
            if (tok.size() < 3) {
                const std::string bound_type = FixedField(line, 2, 3);
                const std::string var = FixedField(line, 15, 22);
                const std::string vals = FixedField(line, 25, 36);
                tok.clear();
                if (!bound_type.empty()) {
                    tok.push_back(bound_type);
                }
                const std::string bname = FixedField(line, 5, 12);
                if (!bname.empty()) {
                    tok.push_back(bname);
                }
                if (!var.empty()) {
                    tok.push_back(var);
                }
                if (!vals.empty()) {
                    tok.push_back(vals);
                }
            }
            if (tok.size() < 3) {
                continue;
            }
            const std::string bound_type = tok[0];
            const std::string var = tok[2];
            const int64_t j = ensure_col(var);
            double val = 0.0;
            if (tok.size() >= 4) {
                ParseDouble(tok[3], &val);
            }
            if (bound_type == "LO") {
                prob.lower_bounds[j] = val;
            } else if (bound_type == "UP") {
                prob.upper_bounds[j] = val;
            } else if (bound_type == "FX") {
                prob.lower_bounds[j] = val;
                prob.upper_bounds[j] = val;
            } else if (bound_type == "FR") {
                prob.lower_bounds[j] = -kBoundInf;
                prob.upper_bounds[j] = kBoundInf;
            } else if (bound_type == "MI") {
                prob.lower_bounds[j] = -kBoundInf;
            } else if (bound_type == "PL") {
                prob.upper_bounds[j] = kBoundInf;
            }
        }
    }
    return prob;
}

StandardFormLp BuildStandardFormLp(const MPSProblem& prob) {
    StandardFormLp lp;
    const int n_orig = static_cast<int>(prob.col_index.size());
    const int m_orig = static_cast<int>(prob.row_index.size());
    if (n_orig == 0) {
        lp.A.resize(0, 0);
        lp.b.resize(0);
        lp.c.resize(0);
        return lp;
    }

    std::vector<double> lo(static_cast<size_t>(n_orig), 0.0);
    std::vector<double> up(static_cast<size_t>(n_orig), kBoundInf);
    std::vector<char> has_lo(static_cast<size_t>(n_orig), 0);
    std::vector<char> has_up(static_cast<size_t>(n_orig), 0);
    for (const auto& entry : prob.lower_bounds) {
        if (entry.first >= 0 && entry.first < n_orig) {
            lo[static_cast<size_t>(entry.first)] = entry.second;
            has_lo[static_cast<size_t>(entry.first)] = 1;
        }
    }
    for (const auto& entry : prob.upper_bounds) {
        if (entry.first >= 0 && entry.first < n_orig) {
            up[static_cast<size_t>(entry.first)] = entry.second;
            has_up[static_cast<size_t>(entry.first)] = 1;
        }
    }
    for (int j = 0; j < n_orig; ++j) {
        if (has_up[static_cast<size_t>(j)] && !has_lo[static_cast<size_t>(j)] &&
            up[static_cast<size_t>(j)] < 0.0) {
            lo[static_cast<size_t>(j)] = -kBoundInf;
        }
    }

    std::vector<double> c_orig(static_cast<size_t>(n_orig), 0.0);
    for (const auto& entry : prob.objective_row_coefficients) {
        if (entry.first >= 0 && entry.first < n_orig) {
            c_orig[static_cast<size_t>(entry.first)] = entry.second;
        }
    }

    std::vector<char> row_type(static_cast<size_t>(m_orig), 'E');
    for (const auto& entry : prob.row_index) {
        auto type_it = prob.row_types.find(entry.first);
        if (type_it != prob.row_types.end() && entry.second >= 0 &&
            entry.second < m_orig) {
            row_type[static_cast<size_t>(entry.second)] = type_it->second;
        }
    }

    std::vector<VarMap> vars(static_cast<size_t>(n_orig));
    std::vector<double> c_out;
    std::vector<std::pair<int, double>> boxed;
    c_out.reserve(static_cast<size_t>(n_orig) * 2);
    double offset = 0.0;

    auto new_col = [&](double cj) {
        const int idx = static_cast<int>(c_out.size());
        c_out.push_back(cj);
        return idx;
    };

    for (int j = 0; j < n_orig; ++j) {
        const double lj = lo[static_cast<size_t>(j)];
        const double uj = up[static_cast<size_t>(j)];
        const double cj = c_orig[static_cast<size_t>(j)];
        VarMap& v = vars[static_cast<size_t>(j)];
        if (!IsNegInf(lj) && !IsPosInf(uj) && std::abs(uj - lj) <= 1e-12) {
            v.shift = lj;
            offset += cj * lj;
            continue;
        }
        if (IsNegInf(lj) && IsPosInf(uj)) {
            v.pos = new_col(cj);
            v.neg = new_col(-cj);
            continue;
        }
        if (!IsNegInf(lj)) {
            v.pos = new_col(cj);
            v.shift = lj;
            v.coeff_sign = 1.0;
            offset += cj * lj;
            if (!IsPosInf(uj)) {
                boxed.emplace_back(v.pos, uj - lj);
            }
            continue;
        }
        v.pos = new_col(-cj);
        v.shift = uj;
        v.coeff_sign = -1.0;
        offset += cj * uj;
    }

    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> trips;
    std::vector<double> b_out;

    auto add_row = [&](const std::vector<std::pair<int, double>>& cols, double rhs) {
        const int row = static_cast<int>(b_out.size());
        for (const auto& entry : cols) {
            if (entry.second != 0.0) {
                trips.emplace_back(row, entry.first, entry.second);
            }
        }
        b_out.push_back(rhs);
    };

    auto transform_row = [&](int orig_row, double* shift_dot) {
        std::vector<std::pair<int, double>> cols;
        *shift_dot = 0.0;
        auto it = prob.coefficients.find(orig_row);
        if (it == prob.coefficients.end()) {
            return cols;
        }
        for (const auto& entry : it->second) {
            if (entry.first < 0 || entry.first >= n_orig) {
                continue;
            }
            const VarMap& v = vars[static_cast<size_t>(entry.first)];
            const double aij = entry.second;
            *shift_dot += aij * v.shift;
            if (v.pos >= 0) {
                cols.emplace_back(v.pos, aij * v.coeff_sign);
            }
            if (v.neg >= 0) {
                cols.emplace_back(v.neg, -aij);
            }
        }
        return cols;
    };

    for (int i = 0; i < m_orig; ++i) {
        double shift_dot = 0.0;
        auto cols = transform_row(i, &shift_dot);
        double rhs_i = 0.0;
        auto rhs_it = prob.rhs.find(i);
        if (rhs_it != prob.rhs.end()) {
            rhs_i = rhs_it->second;
        }
        const bool has_range = prob.ranges.find(i) != prob.ranges.end();
        const double range = has_range ? prob.ranges.at(i) : 0.0;
        double lb = 0.0;
        double ub = 0.0;
        ConstraintBounds(row_type[static_cast<size_t>(i)], rhs_i, has_range, range, &lb,
                         &ub);
        lb -= shift_dot;
        ub -= shift_dot;

        const bool finite_lb = !IsNegInf(lb);
        const bool finite_ub = !IsPosInf(ub);
        if (finite_lb && finite_ub && std::abs(ub - lb) <= 1e-12) {
            add_row(cols, lb);
            continue;
        }
        if (finite_ub) {
            auto cols_u = cols;
            cols_u.emplace_back(new_col(0.0), 1.0);
            add_row(cols_u, ub);
        }
        if (finite_lb) {
            auto cols_l = cols;
            cols_l.emplace_back(new_col(0.0), -1.0);
            add_row(cols_l, lb);
        }
    }

    for (const auto& box : boxed) {
        std::vector<std::pair<int, double>> cols;
        cols.emplace_back(box.first, 1.0);
        cols.emplace_back(new_col(0.0), 1.0);
        add_row(cols, box.second);
    }

    const int m = static_cast<int>(b_out.size());
    const int n = static_cast<int>(c_out.size());
    lp.A.resize(m, n);
    lp.A.setFromTriplets(trips.begin(), trips.end());
    lp.A.makeCompressed();
    lp.b = Eigen::Map<const Eigen::VectorXd>(b_out.data(), m);
    lp.c = Eigen::Map<const Eigen::VectorXd>(c_out.data(), n);
    lp.objective_offset = offset;
    return lp;
}

void BuildDenseLp(const MPSProblem& prob, Eigen::MatrixXd* A, Eigen::VectorXd* b,
                  Eigen::VectorXd* c) {
    StandardFormLp lp = BuildStandardFormLp(prob);
    *A = Eigen::MatrixXd(lp.A);
    *b = lp.b;
    *c = lp.c;
}

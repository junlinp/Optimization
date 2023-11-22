#ifndef SPARSE_MATRIX_H_
#define SPARSE_MATRIX_H_

#include <vector>
#include "Eigen/Dense"

class SparseMatrix {
public:
    SparseMatrix (const Eigen::MatrixXd& A) : row_(0), col_(0) {
        int row = A.rows();
        int col = A.cols();
        row_ = row;
        col_ = col;
        row_idx.resize(row + 1);

        for(int r = 0; r < row; r++) {
            row_idx[r] = value.size();
            for (int c = 0; c < col; c++) {
                double v = A(r, c);
                if ( std::abs(v) >= 1e-9) {
                    value.push_back(A(r, c));
                    col_idx.push_back(c);
                }
            }
        }
        row_idx[row] = value.size();
    }

    int nnz() {
        return value.size();
    }

    int rows() {
        return row_;
    }

    int cols() {
        return col_;
    }

    int row_, col_;
    std::vector<double> value;
    std::vector<int> col_idx;
    std::vector<int> row_idx;
};
#endif  // SPARSE_MATRIX_H_
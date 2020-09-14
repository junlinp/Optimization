#include "gtest/gtest.h"
#include "stdio.h"
#include "string.h"
#include "Eigen/Dense"
#include "chrono"
#include "sparse_matrix.h"

extern "C" {
#include "mmio.h"
}


void ReadMatrix(const std::string& file_name, Eigen::MatrixXd& matrix) {
    MM_typecode mattype;
    FILE* fp = fopen(file_name.c_str(), "r");
    ASSERT_TRUE(fp != NULL);
    mm_read_banner(fp, &mattype);
    int row, col, nz;
    mm_read_mtx_crd_size(fp, &row, &col, &nz);
    ASSERT_GT(row, 0);
    ASSERT_GT(col, 0);
    ASSERT_GT(nz, 0);
    matrix = Eigen::MatrixXd(row, col);
    for (int i = 0; i < row * col; i++) {
        matrix.data()[i] = 0.0;
    }
    int count = 0;
    for(int i = 0; i < nz; i++) {
        int r, c;
        double v;
        fscanf(fp, "%d %d %lf", &r, &c, &v); 
        matrix(r - 1, c - 1) = v;
        if (std::abs(v) >= std::numeric_limits<double>::epsilon()) {
            count++;
        }
    }
    fclose(fp);
}

void ReadVector(const std::string& file_name, Eigen::VectorXd& vector) {
    MM_typecode mattype;
    FILE* fp = fopen(file_name.c_str(), "r");
    ASSERT_TRUE(fp != NULL);
    mm_read_banner(fp, &mattype);
    int row, col;
    mm_read_mtx_array_size(fp, &row, &col);
    assert(col == 1);
    vector = Eigen::VectorXd::Zero(row);
    for (int i = 0; i < row; i++) {
        double v;
        fscanf(fp, "%lf", &v);
        vector(i) = v;
    }
}

class TimeUse {

public:
    TimeUse() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    ~TimeUse() {
        std::chrono::milliseconds duration = 
            std::chrono::duration_cast<std::chrono::milliseconds>  (std::chrono::high_resolution_clock::now() - start_);
        std::cout << "Time excaped :" << duration.count() << " milliseconds" << std::endl;
    }

private:
    std::chrono::steady_clock::time_point start_;
};

void EigenSolver(const Eigen::MatrixXd & A, const Eigen::VectorXd& b, Eigen::VectorXd& x) {
    TimeUse t;
    x = A.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);
    std::cout << "Eigen jacobi Svd ";
}

TEST(illc1850, Sparse_Matrix_IO) {
    std::string file_name = std::string(Test_DIR) + "/illc1850.mtx";
    std::string b_file_name = std::string(Test_DIR) +"/illc1850_rhs1.mtx";
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    ReadMatrix(file_name, A);
    ReadVector(b_file_name, b);
    Eigen::VectorXd x;
    SparseMatrix sA(A);
    EXPECT_EQ(sA.nnz(), 8636);
    EigenSolver(A, b, x);
}

int main() {
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
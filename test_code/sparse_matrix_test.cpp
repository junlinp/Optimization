#include "gtest/gtest.h"
#include "stdio.h"
#include "string.h"
#include "Eigen/Dense"
#include <chrono>
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
        std::cout << "Time Elapsed :" << duration.count() << " milliseconds" << std::endl;
    }
private:
    decltype(std::chrono::high_resolution_clock::now()) start_;
};

void InverseSolver(const Eigen::MatrixXd & A, const Eigen::VectorXd& b, Eigen::VectorXd& x) {
    TimeUse t;
    x = A.inverse() * b;
    std::cout << "Inverse ";
}

void EigenSolver(const Eigen::MatrixXd & A, const Eigen::VectorXd& b, Eigen::VectorXd& x) {
    TimeUse t;
    x = A.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);
    std::cout << "Eigen jacobi Svd ";
}

void FOM(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::VectorXd& x, int max_num = 32) {
    TimeUse t;
    if (A.rows() != A.cols()) {
        printf("Warning: A is not symmetric\n");
        return;
    }

    Eigen::VectorXd r0 = b - A * x;
    double beta = r0.norm();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(A.rows(), A.cols());
    int n = A.rows(); 
    std::vector<Eigen::VectorXd> v_ = {r0 / beta};
    int bound = std::min(n, max_num);
    for(int j = 0; j < bound; j++) {
        //std::cout << "j : " << j << std::endl;
        Eigen::VectorXd w = A * v_[j];
        for (int i = 0; i <= j; i++) {
            H(i, j) = w.dot(v_[i]);
            w = w - H(i, j) * v_[i];
        }
        double h_j_1_j = w.norm();
        if ( std::abs(h_j_1_j) < std::numeric_limits<double>::epsilon()) {
            break;
        }
        if (j < bound - 1) {
            v_.push_back(w / h_j_1_j);
            H(j + 1, j) = h_j_1_j;
        }
    }
    int m = v_.size();
    //std::cout << "m : " << m << std::endl;
    Eigen::MatrixXd H_ = H.block(0, 0, m, m);
    Eigen::MatrixXd V(n, m);
    for(int i = 0; i < m ; i++) {
        V.col(i) = v_[i];
    }
    Eigen::VectorXd e1(m, 1);
    for(int i = 0; i < m; i++) {
        e1(i) = 0.0;
    }
    e1(0) = 1;
    x = x + V * H_.inverse() * beta * e1;
    std::cout << "FMO ";
}

void FOM_v1(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::VectorXd& x) {

    for(int i = 0; i < 128; i++) {
        FOM(A, b, x, i + 1);
    }
}

void JacobianIterativeSolver(const Eigen::MatrixXd &A, const Eigen::VectorXd& b, Eigen::VectorXd& x) {
    TimeUse t;
    Eigen::MatrixXd D = A;
    Eigen::MatrixXd E_plus_F = A;
    for(int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            if (i != j) {
                D(i, j) = 0.0;
            } else {
                E_plus_F(i, j) = 0.0;
                D(i, j) = 1.0 / (A(i, j) + std::numeric_limits<double>::epsilon());
            }
        }
    }
    //std::cout << D << std::endl;
    const size_t max_iterator = 16;
    Eigen::MatrixXd D_inverse = D.inverse();
    //std::cout << "D_inverse norm :" << D_inverse.norm() << std::endl;
    Eigen::MatrixXd D_inverse_mul_E_plus_F = D_inverse * E_plus_F;
    Eigen::MatrixXd D_inverse_mul_b = D_inverse * b;
    std::cout << "G Norm : " << D_inverse_mul_E_plus_F.norm() << std::endl;
    for (size_t i = 0; i < max_iterator; i++) {
        //std::cout << "x normal : " << x.norm() << std::endl;
        x = D_inverse_mul_E_plus_F * x + D_inverse_mul_b;
    }
    std::cout << "jacobi Solver ";

}
TEST(illc1850, InverseSolver) {
    std::string file_name = std::string(Test_DIR) + "/illc1850.mtx";
    std::string b_file_name = std::string(Test_DIR) +"/illc1850_rhs1.mtx";
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    ReadMatrix(file_name, A);
    ReadVector(b_file_name, b);
    Eigen::VectorXd x;
    //SparseMatrix sA(A);
    //EXPECT_EQ(sA.nnz(), 8636);
    InverseSolver(A.transpose() * A, A.transpose() * b, x);
    Eigen::VectorXd residuals = b - A * x;
    std::cout << "RMSE : " << residuals.norm() << std::endl;
}
/*
TEST(illc1850, bdcSolver) {
    std::string file_name = std::string(Test_DIR) + "/illc1850.mtx";
    std::string b_file_name = std::string(Test_DIR) +"/illc1850_rhs1.mtx";
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    ReadMatrix(file_name, A);
    ReadVector(b_file_name, b);
    Eigen::VectorXd x;
    //SparseMatrix sA(A);
    //EXPECT_EQ(sA.nnz(), 8636);
    EigenSolver(A.transpose() * A, A.transpose() * b, x);
    Eigen::VectorXd residuals = b - A * x;
    std::cout << "RMSE : " << residuals.norm() << std::endl;
}
*/

TEST(illc1850, FOM) {
    std::string file_name = std::string(Test_DIR) + "/illc1850.mtx";
    std::string b_file_name = std::string(Test_DIR) +"/illc1850_rhs1.mtx";
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    ReadMatrix(file_name, A);
    ReadVector(b_file_name, b);
    //std::cout << (A.transpose() * A).norm() << std::endl;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(A.cols());
    //SparseMatrix sA(A);
    //EXPECT_EQ(sA.nnz(), 8636);
    FOM(A.transpose() * A, A.transpose() * b, x);

    Eigen::VectorXd residuals = b - A * x;
    std::cout << "RMSE : " << residuals.norm() << std::endl;
}

TEST(illc1850, FOM_v1) {
    std::string file_name = std::string(Test_DIR) + "/illc1850.mtx";
    std::string b_file_name = std::string(Test_DIR) +"/illc1850_rhs1.mtx";
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    ReadMatrix(file_name, A);
    ReadVector(b_file_name, b);
    //std::cout << (A.transpose() * A).norm() << std::endl;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(A.cols());
    //SparseMatrix sA(A);
    //EXPECT_EQ(sA.nnz(), 8636);
    FOM_v1(A.transpose() * A, A.transpose() * b, x);

    Eigen::VectorXd residuals = b - A * x;
    std::cout << "RMSE : " << residuals.norm() << std::endl;
}

TEST(FOM, Simple_Case) {
    Eigen::MatrixXd A(2, 2);
    A << 1, 1,
        5, 7;
    Eigen::VectorXd b(2);
    b << 4, 26;

    Eigen::VectorXd x(2);
    x << 0.0, 0.0;
    FOM(A, b, x);

    std::cout << "x : " << x << std::endl;
}
/*
TEST(illc1850, JacobianIterativeSolver) {
    std::string file_name = std::string(Test_DIR) + "/illc1850.mtx";
    std::string b_file_name = std::string(Test_DIR) +"/illc1850_rhs1.mtx";
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    ReadMatrix(file_name, A);
    ReadVector(b_file_name, b);
    //std::cout << (A.transpose() * A).norm() << std::endl;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(A.cols());
    //SparseMatrix sA(A);
    //EXPECT_EQ(sA.nnz(), 8636);
    JacobianIterativeSolver(A.transpose() * A, A.transpose() * b, x);

    Eigen::VectorXd residuals = b - A * x;
    std::cout << "RMSE : " << residuals.norm() << std::endl;
}
*/

int main() {
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}

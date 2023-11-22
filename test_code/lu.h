#include <Eigen/Dense>

inline Eigen::VectorXd ComputeWk(const Eigen::MatrixXd& X, size_t k) {
    size_t n = X.rows();
    //size_t m = X.cols();
    double beta = 0.0;
    for (size_t i = k; i < n; i++) {
        beta += X(i, k) * X(i, k);
    }
    beta = std::sqrt(beta) * X(k, k) / std::abs(X(k, k));

    Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
    for(size_t i = k; i < n; ++i) {
        z(i) = X(i, k);
        if (i == k) {
            z(i) += beta;
        }
    }
    //std::cout << "z: " << z << std::endl;
    return z.normalized();
}
void HouseHolder(const Eigen::MatrixXd& X, Eigen::MatrixXd& Q, Eigen::MatrixXd& R) {
    size_t n = X.rows();
    size_t m = X.cols();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(n, n);
    Q(0, 0) = 0;
    for (size_t k = 0; k < m; k++) {
        Eigen::VectorXd w = ComputeWk(P * X, k);
        //std::cout << "W" << k << ": " << w << std::endl;
        Eigen::MatrixXd P_k = I - 2 * w * w.transpose();
        //std::cout << "P" << k << " : " << P_k << std::endl;
        P = P_k * P;
        //std::cout << "P" << " : " << P << std::endl;
        //std::cout << "X" << k << " : " << X.col(k) << std::endl;
        Eigen::VectorXd r_k = P * X.col(k);
        R.col(k) = r_k;
    }
    Q = P;
}


void GMRES(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::VectorXd& initial_x) {
    Eigen::VectorXd r0 = A * initial_x - b;
    size_t n = A.rows();
    size_t m = A.cols();
    Eigen::MatrixXd V(n, m);
    V.col(0) = r0 / r0.norm();
    // double beta = r0.norm();
    Eigen::MatrixXd H(m + 1, m);
    for(size_t i = 0; i < m - 1; i++) {
        Eigen::VectorXd w = A * V.col(i);
        for(size_t j = 0; j <= i; j++) {
            H(j, i) = w.dot(V.col(j));
            w = w - H(j, i) * V.col(j);
        }
        H(i + 1, i) = w.norm();
        V.col(i + 1) = w / w.norm();
    }
    std::cout << "H : " << H << std::endl;
    // solve the || beta * e_1 - H * y ||_2
    
}
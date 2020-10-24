#include <Eigen/Dense>

inline Eigen::VectorXd ComputeWk(const Eigen::MatrixXd& X, size_t k) {
    size_t n = X.rows();
    //size_t m = X.cols();
    double beta = 0.0;
    for (size_t i = k; i < n; i++) {
        beta += X(i, k) * X(i, k);
    }
    beta = std::sqrt(beta);
    beta *= X(k, k) / std::abs(X(k, k));

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
        Eigen::VectorXd w = ComputeWk(X, k);
        //std::cout << "W" << k << ": " << w << std::endl;
        Eigen::MatrixXd P_k = I - 2 * w * w.transpose();
        //std::cout << "P" << k << " : " << P_k << std::endl;
        P = P_k * P;
        Eigen::VectorXd r_k = P * X.col(k);
        R.col(k) = r_k;
    }
    Q = P;
}
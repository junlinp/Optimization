#include <Eigen/Dense>

double ComputeClusterError(Eigen::MatrixXd& K, Eigen::MatrixXd& W, Eigen::VectorXi& base_cluster, size_t node_id, size_t cluster_id) {
    double res = K(node_id, node_id);
    double res_1 = 0.0;
    double res_1_sum = 0.0;
    for(int i = 0; i < base_cluster.size(); i++) {
        if (base_cluster(i) == cluster_id) {
            res_1 += 2 * W(i, i) * K(node_id, j);
            res_1_sum = W(i, i);
        }
    }
    res_1 /= res_1_sum;
    double res_2 = 0.0;
    for(int i = 0; i < base_cluster.size(); i++) {
        for (int j = 0; j < base_cluster.size(); j++) {
            if (base_cluster(i) == cluster_id && base_cluster(j) == cluster_id) {
                res_2 += W(i, i) * W(j, j) * K(i, j);
            }
        }
    }
    res_2 /= res_1_sum * res_1_sum;
    return res - res_1 + res_2;
}
template <int CLUSTER_SIZE>
void k_mean(Eigen::MatrixXd K, Eigen::MatrixXd W, Eigen::VectoriD indicator) {
  int max_iterator = 1024;
  int iterator = 0;
  while (iterator++ < max_iterator) {
    Eigen::VectoriD temp_indicator = indicator;
    for (int i = 0; i < indicator.size(); i++) {
      double cluster_error = std::numeric_limits<double>::max();
      for (int j = 0; j < CLUSTER_SIZE; j++) {
        double c = ComputeClusterError(K, W, i, j);
        if (cluster_error > c) {
          cluster_error = c;
          temp_indicator(i) = j;
        }
      }
      std::swap(indicator, temp_indicator);
    }
  }
}
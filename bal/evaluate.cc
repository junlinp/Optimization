#include "evaluate.h"

#include <memory>

#include "../linear_solver.h"
#include "iostream"
void Evaluate(const Problem& problem, Eigen::VectorXd& error,
              Eigen::SparseMatrix<double>& jacobian) {
  size_t observation_size = problem.observations_.size();
  size_t camera_size = problem.cameras_.size();
  size_t point_size = problem.points_.size();
  error = Eigen::VectorXd::Zero(2 * observation_size);
  jacobian = Eigen::SparseMatrix<double>(2 * observation_size,
                                         camera_size * 9 + point_size * 3);
  size_t count = 0;
  using Triple = Eigen::Triplet<double>;
  std::vector<Triple> reserver_triple;
  reserver_triple.reserve(2 * observation_size * 12);
  for (const auto& pair : problem.observations_) {
    size_t camera_ids = pair.first.first;
    size_t point_ids = pair.first.second;
    Observation o = pair.second;
    ProjectFunction pf(o(0), o(1));
    Jet<9 + 3> X[12];
    CameraParam c = problem.cameras_.at(camera_ids);
    Landmark p = problem.points_.at(point_ids);
    X[0] = Jet<12>(c.params[0], 0);
    X[1] = Jet<12>(c.params[1], 1);
    X[2] = Jet<12>(c.params[2], 2);
    X[3] = Jet<12>(c.params[3], 3);
    X[4] = Jet<12>(c.params[4], 4);
    X[5] = Jet<12>(c.params[5], 5);
    X[6] = Jet<12>(c.params[6], 6);
    X[7] = Jet<12>(c.params[7], 7);
    X[8] = Jet<12>(c.params[8], 8);
    X[9] = Jet<12>(p(0), 9);
    X[10] = Jet<12>(p(1), 10);
    X[11] = Jet<12>(p(2), 11);
    Jet<12> residual[2];
    pf(X, X + 9, residual);
    // std::cout << "Residual : " << residual[0].value() << " , " <<
    // residual[1].value() << std::endl;
    error(2 * count + 0) = residual[0].value();
    error(2 * count + 1) = residual[1].value();
    for (size_t j = 0; j < 9; j++) {
      /*
    jacobian.insert(2 * count + 0, 9 * camera_ids + j) =
        residual[0].Gradient()(j);
    jacobian.insert(2 * count + 1, 9 * camera_ids + j) =
        residual[1].Gradient()(j);
    */
      reserver_triple.push_back(
          Triple(2 * count + 0, 9 * camera_ids + j, residual[0].Gradient()(j)));
      reserver_triple.push_back(
          Triple(2 * count + 1, 9 * camera_ids + j, residual[1].Gradient()(j)));
    }

    for (size_t j = 0; j < 3; j++) {
      /*
    jacobian.insert(2 * count + 0, 9 * camera_size + point_ids * 3 + j) =
        residual[0].Gradient()(9 + j);
    jacobian.insert(2 * count + 1, 9 * camera_size + point_ids * 3 + j) =
        residual[1].Gradient()(9 + j);
      */
      reserver_triple.push_back(Triple(2 * count + 0,
                                       9 * camera_size + point_ids * 3 + j,
                                       residual[0].Gradient()(9 + j)));
      reserver_triple.push_back(Triple(2 * count + 1,
                                       9 * camera_size + point_ids * 3 + j,
                                       residual[1].Gradient()(9 + j)));
    }
    count++;
  }
  jacobian.setFromTriplets(reserver_triple.begin(), reserver_triple.end());
}

void UpdateStep(Problem& problem, Eigen::VectorXd& step) {
    size_t camera_size = problem.cameras_.size();
    size_t point_size = problem.points_.size();
    std::map<size_t, std::vector<double>> camera_step;
    std::map<size_t, std::vector<double>> point_step;
    for (size_t camera_id = 0; camera_id < camera_size; camera_id++) {
        std::vector<double> t(9);
        for(size_t j = 0; j < 9; j++) {
            t[j] = step(9 * camera_id + j);
        }
        camera_step[camera_id] = t;
    }

    for (size_t point_id = 0; point_id < point_size; point_id++) {
        std::vector<double> t(3);
        for(size_t j = 0; j < 3; j++) {
            t[j] = step(9 * camera_size + point_id * 3 + j);
        }
        point_step[point_id] = t;
    }

    problem.Update(camera_step, point_step);
}

void LM(Problem& problem) {
  Eigen::VectorXd f;
  Eigen::SparseMatrix<double> jacobian;
  Evaluate(problem, f, jacobian);
  size_t col = jacobian.cols();
  double lambda = 0.0;
  for (size_t i = 0; i < col; i++) {
      double norm = jacobian.col(i).norm();
      lambda = std::max(lambda, norm);
  }
  double threshold = 0.75;

  std::cout << "Evaluate Finish" << std::endl;
  std::cout << "Inital RMS : " << f.norm() << std::endl;
  size_t max_iterator = 50;
  size_t iterator = 0;
  lambda = 1;
  while(iterator++ < max_iterator) {
    Evaluate(problem, f, jacobian);
    //std::cout << "Jacobian : " << jacobian << std::endl;
    Eigen::VectorXd b = -jacobian.transpose() * f;

    Eigen::SparseMatrix<double> D(col, col);
    for (size_t i = 0; i < col; i++) {
      double norm = jacobian.col(i).norm();
      D.insert(i, i) = norm;
    }
    std::shared_ptr<AbstratorCoefficient> A =
        std::make_shared<BundleAdjustmentNormalFormulaCoefficient<double>>(jacobian, D, lambda);
    Eigen::VectorXd step;
    ConjugateGradient(A, b, step);
    auto residual = b - A->Multiple(step);
    std::cout << "residual : " << residual.norm() << std::endl;
    //std::cout << "step : " << step << std::endl;
    //Eigen::MatrixXd jacobian2 = jacobian;
    //Eigen::MatrixXd D2 = D;
    //Eigen::VectorXd step2 = (jacobian2.transpose() * jacobian2 + lambda * D2.transpose() * D2).inverse() * b;
    //std::cout << "step2 : " << step2 << std::endl;
    Problem update_problem = problem;
    UpdateStep(update_problem, step);
    Eigen::VectorXd update_f;
    Eigen::SparseMatrix<double> update_jacobian;
    Evaluate(update_problem, update_f, update_jacobian);
    //std::cout << "Update_f : " << update_f << std::endl;
    double origin_error = 0.5 * f.dot(f) / f.rows();
    double update_error = 0.5 * update_f.dot(update_f) / f.rows();
    double function_decrese = origin_error - update_error;
    auto temp = f + jacobian * step;
    double estimate_error = 0.5 * temp.dot(temp) / temp.rows();
    double estimate_decres = origin_error - estimate_error;

    std::cout << "Origin Error : " << origin_error << std::endl;
    std::cout << "Update Error : " << update_error << std::endl;
    std::cout << "Estimate_error : " << estimate_error << std::endl;
    std::cout << "lambda : " << lambda << std::endl;
    if (function_decrese / estimate_decres > threshold) {
        UpdateStep(problem, step);
        lambda = std::max(lambda / 1.1, 1e-7);
        std::cout << "Update the Parameter " << std::endl;
    } else {
        lambda *= 1.1; 
    }

  }

  std::cout << "LM RMS : " << f.norm() << std::endl;
}
docker run --privileged -i -t --rm --volumes-from ikev2-vpn-server -e "HOST=209.250.245.34" gaomd/ikev2-vpn-server:0.3.0 generate-mobileconfig > ikev2-vpn.mobileconfig
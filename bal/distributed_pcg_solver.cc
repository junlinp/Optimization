#include "distributed_pcg_solver.h"
#include "Eigen/Sparse"
#include "cost_function_auto.h"
#include "problem.h"
#include <Eigen/src/IterativeLinearSolvers/ConjugateGradient.h>
#include <Eigen/src/SparseCholesky/SimplicialCholesky.h>
#include <ceres/cost_function.h>

#include <limits>
#include <map>
#include <set>
#include <utility>
template <class JC, class JP, class V, class X>
Eigen::VectorXd MatrixPlusX(const JC& jc, const JP& jp, const V& v, const X& x) {
  Eigen::VectorXd jc_mul_x = jc * x;

  Eigen::VectorXd jc_t_jc_mul_x = jc.transpose() * jc_mul_x;
  auto llt_solve = Eigen::SimplicialLLT<V>();
  llt_solve.compute(v); 
  Eigen::VectorXd t = llt_solve.solve(jp.transpose() * jc_mul_x);
  t = jp * t;
  t = jc.transpose() * t;
  return jc_t_jc_mul_x - t;
}

template<class JC, class JP, class V, class X, class B>
X CG(const JC& jc, const JP& jp, const V& v, const B& b, const X& x0) {
  Eigen::VectorXd p = b - MatrixPlusX(jc, jp, v, x0);
  auto r = p;
  X x = x0;
  for (int k = 0; k < x0.rows(); k++) {
    auto norm_rk = r.dot(r);
    auto Apk = MatrixPlusX(jc, jp, v, p);
    auto norm_p0 = p.dot(Apk);
    x += norm_rk / norm_p0 * p;
    r = b - MatrixPlusX(jc, jp, v, x);
    p = r + norm_rk / r.dot(r) * p;
    std::cout << k << "CG residuals : " << r.norm() << std::endl;
  }
  return x;
}

Eigen::VectorXd PCG(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
  Eigen::VectorXd x0(b.rows());
  x0.setRandom();

  Eigen::MatrixXd M = A.diagonal().asDiagonal();
  std::cout << "M " << M.rows() << " : " << M.cols() << std::endl;
  Eigen::VectorXd r0 = b - A * x0;
  Eigen::VectorXd p = M.llt().solve(r0);
  double r_dot_z = r0.dot(p);
  Eigen::VectorXd w = A * p;
  double alpha = r0.dot(p) / p.dot(w);
  Eigen::VectorXd x = x0 + alpha * p;
  Eigen::VectorXd r = r0 - alpha * w;
  int k = 1;
  int max_k = 1024 * 1024;
  while (r.squaredNorm() > std::numeric_limits<double>::min() && k < max_k) {
    Eigen::VectorXd z = M.llt().solve(r);
    double new_r_dot_z = r.dot(z);
    double beta =  new_r_dot_z / r_dot_z;
    r_dot_z = new_r_dot_z;
    p = z + beta * p;
    w = A * p;
    alpha = r_dot_z / p.dot(w);
    x += alpha * p;
    r = r - alpha * w;
    k++;
  }
  std::cout << "PCG k " << k << std::endl; 
  return x;
}

Eigen::VectorXd CG(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd x0) {
  Eigen::VectorXd x = x0;
  x.setRandom();
  Eigen::VectorXd p = b - A * x;
  Eigen::VectorXd r = p;
  size_t max_k = 1;
  double last_norm_r = r.dot(r);
  double norm_r = last_norm_r;
  for (size_t k = 0; k < max_k; k++) {
    double norm_p = p.dot(A * p);
    if (norm_p < std::numeric_limits<double>::epsilon()) {
      break;
    }
    // std::cout << "move : " << (norm_r / norm_p * p).norm() << std::endl;
    x = x + norm_r / norm_p * p;
    r = b - A * x;
    last_norm_r = norm_r;
    norm_r = r.dot(r);
    p = r + last_norm_r / norm_r * p;

    if (r.norm() < 1e-6) {
      break;
    }
    std::cout << k << " norm_r : " << norm_r << std::endl;
  }
  std::cout << "CG residuals : " << r.norm() << std::endl;
// fill A and b
  Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower | Eigen::Upper> cg;
  cg.compute(A);
  cg.solve(b);
  std::cout << "#iterations:     " << cg.iterations() << std::endl;
  std::cout << "estimated error: " << cg.error() << std::endl;
  // update b, and solve again
  return cg.solve(b);
  return x;
}


template<class Jc, class Jp, class R, class DC, class DP>
void SolveNormalFormula(const Jc& jc, const Jp& jp, const R& r, double mu, DC& delta_c, DP& delta_p) {

  using SM = Eigen::SparseMatrix<double>;
  std::cout << "B" << std::endl;
  SM I_c = SM(jc.cols(), jc.cols());
  SM I_p = SM(jp.cols(), jp.cols());
  I_c.setIdentity();
  I_p.setIdentity();

  SM U = SM(jc.transpose()) * jc + mu * I_c; 
  SM W = SM(jc.transpose()) * jp;
  SM V = SM(jp.transpose()) * jp + mu * I_p;
  
  std::cout << "Jc : " << jc.rows() << " , " << jc.cols() << std::endl;
  std::cout << "Jp : " << jp.rows() << " , " << jp.cols() << std::endl;

  auto llt_solve = Eigen::SimplicialLLT<SM>();
  llt_solve.compute(V);


  SM jpt = jp.transpose();
  Eigen::VectorXd jpT_r = jp.transpose() * r;

  Eigen::MatrixXd g = -jc.transpose() * r + W * llt_solve.solve(jpT_r);


  Eigen::MatrixXd A = U - W * llt_solve.solve(Eigen::MatrixXd(W.transpose()));

  delta_c = Eigen::VectorXd(jc.cols());
  delta_c.setRandom();
  //delta_c = CG(jc, jp, V, g, delta_c);
  Eigen::VectorXd t = A.llt().solve(g);
  std::cout << "b - Ax : " << (g - A * t).norm() << std::endl;
  delta_c = CG(A, g, delta_c);
  std::cout << "CG b - Ax : " << (g - A * delta_c).norm() << std::endl;

  //delta_c = PCG(A, g);
  //std::cout << "PCG b - Ax : " << (g - A * delta_c).norm() << std::endl;

  std::cout << "r norm : " << r.norm() << std::endl;
  delta_p =
      -llt_solve.solve( jpT_r + W.transpose() * delta_c);

  std::cout << "delta_c norm : " << delta_c.norm() << std::endl;
  std::cout << "delta_p norm : " << delta_p.norm() << std::endl;

}




void DistributedPCGSolver::Solve(Problem &problem) {
  using SM = Eigen::SparseMatrix<double>;

  std::map<size_t, size_t> camera_idx_to_Jc_idx;
  std::map<size_t, size_t> point_idx_to_Jp_idx;

  using Triple = Eigen::Triplet<double>;
  std::vector<Triple> jc_triple_vector;
  std::vector<Triple> jp_triple_vector;

  size_t row_count = 0;

  Eigen::VectorXd r(problem.observations_.size() * 2);
  Eigen::VectorXd camera_variable(problem.cameras_.size() * 6);
  Eigen::VectorXd point_variable(problem.points_.size() * 3);

  std::vector<double*> jacobian_cameras;  
  std::vector<double*> jacobian_points;

  std::vector<std::function<void()>> evaluate_functions;
  std::vector<std::function<void()>> jc_jp_functions;
  // initial variable

  for (const auto& [edge, observation] : problem.observations_) {
    size_t camera_idx = edge.first;
    size_t point_idx = edge.second;
    if (camera_idx_to_Jc_idx.find(camera_idx) == camera_idx_to_Jc_idx.end()) {
      camera_idx_to_Jc_idx.insert(
          std::make_pair(camera_idx, camera_idx_to_Jc_idx.size()));
    }
    if (point_idx_to_Jp_idx.find(point_idx) == point_idx_to_Jp_idx.end()) {
      point_idx_to_Jp_idx.insert(
          std::make_pair(point_idx, point_idx_to_Jp_idx.size()));
    }
    const CameraParam &camera_parameters = problem.cameras_.at(camera_idx);
    const Landmark &point = problem.points_.at(point_idx);
    std::copy(camera_parameters.data(), camera_parameters.data() + ProjectFunction::CAMERA_PARAMETER_SIZE, camera_variable.data() + 6 * camera_idx_to_Jc_idx[camera_idx]);
    std::copy(point.data(), point.data() + 3, point_variable.data() + 3 * point_idx_to_Jp_idx[point_idx]);
  }

  for (const auto &[edge, observation] : problem.observations_) {
    size_t camera_idx = edge.first;
    size_t point_idx = edge.second;
    const CameraParam &camera_parameters = problem.cameras_.at(camera_idx);
    const Landmark &point = problem.points_.at(point_idx);
    std::copy(camera_parameters.data(), camera_parameters.data() + 6, camera_variable.data() + 6 * camera_idx_to_Jc_idx[camera_idx]);
    std::copy(point.data(), point.data() + 3, point_variable.data() + 3 * point_idx_to_Jp_idx[point_idx]);
    const double *camera_intrinsics = camera_parameters.data() + 6;

    jacobian_cameras.push_back(new double[2 * ProjectFunction::CAMERA_PARAMETER_SIZE]);
    jacobian_points.push_back(new double[2 * ProjectFunction::POINT_PARAMETER_SIZE]);

    auto evaluate_functor = [f = camera_intrinsics[0], cx = camera_intrinsics[1],
                    cy = camera_intrinsics[2], u = observation.u(),
                    v = observation.v(),
                    camera_parameters_ptr =
                        camera_variable.data() +
                        6 * camera_idx_to_Jc_idx[camera_idx],
                    point_ptr = point_variable.data() +
                                3 * point_idx_to_Jp_idx[point_idx],
                    jacobian_camera = jacobian_cameras[row_count],
                    jacobian_point = jacobian_points[row_count],
                    residuals_ptr = r.data() + row_count * 2]() {
      ceres::CostFunction *cost_function_ptr =
          ProjectFunction::CreateCostFunction(f, cx, cy, u, v);
      std::vector<double *> jacobians = {jacobian_camera, jacobian_point};
      std::vector<const double *> parameters = {camera_parameters_ptr,
                                                point_ptr};
      cost_function_ptr->Evaluate(parameters.data(), residuals_ptr, jacobians.data());

      delete cost_function_ptr;
    };
    evaluate_functions.push_back(std::move(evaluate_functor));

    auto jc_jp_construction = [jc_offset =
                                   camera_idx_to_Jc_idx[camera_idx] *
                                   ProjectFunction::CAMERA_PARAMETER_SIZE,
                               jp_offset =
                                   point_idx_to_Jp_idx.at(point_idx) *
                                   ProjectFunction::POINT_PARAMETER_SIZE,
                               jacobian_camera = jacobian_cameras[row_count],
                               jacobian_point = jacobian_points[row_count],
                               rw = row_count, &jc_triple_vector,
                               &jp_triple_vector]() {
      for (int k = 0; k < 2; k++) {
        for (int i = 0; i < ProjectFunction::CAMERA_PARAMETER_SIZE; i++) {
          Triple t{
              2 * rw + k, jc_offset + i,
              jacobian_camera[k * ProjectFunction::CAMERA_PARAMETER_SIZE + i]};
          jc_triple_vector.push_back(t);
        }

        for (int i = 0; i < ProjectFunction::POINT_PARAMETER_SIZE; i++) {
          Triple t{
              2 * rw + k, jp_offset + i,
              jacobian_point[k * ProjectFunction::POINT_PARAMETER_SIZE + i]};
          jp_triple_vector.push_back(t);
        }
      }
    };
    jc_jp_functions.push_back(std::move(jc_jp_construction));
    row_count++;
  }

  size_t rows = problem.observations_.size() * 2;
  size_t col_of_jc = camera_idx_to_Jc_idx.size() * ProjectFunction::CAMERA_PARAMETER_SIZE;
  size_t col_of_jp = point_idx_to_Jp_idx.size() * ProjectFunction::POINT_PARAMETER_SIZE;
  SM Jc(rows, col_of_jc);
  SM Jp(rows, col_of_jp);

  for (auto& f : evaluate_functions) {
    f();
  }

  for (auto& f : jc_jp_functions) {
    f();
  }
  Jc.setFromTriplets(jc_triple_vector.begin(), jc_triple_vector.end());
  Jp.setFromTriplets(jp_triple_vector.begin(), jp_triple_vector.end());
  std::cout << "jc nnz : " << Jc.nonZeros() << std::endl;
  std::cout << "jp nnz : " << Jp.nonZeros() << std::endl;
  jc_triple_vector.clear();
  jp_triple_vector.clear();

  // Build jacobian

  double mu = 5.0;
  double v = 1.0;

  size_t epoch = 0;
  size_t max_epoches = 16;
  double epsilon = 1e-6;

  while(epoch++ < max_epoches) {
    for (auto &f : evaluate_functions) {
      f();
    }

    for (auto &f : jc_jp_functions) {
      f();
    }
    Jc.setFromTriplets(jc_triple_vector.begin(), jc_triple_vector.end());
    Jp.setFromTriplets(jp_triple_vector.begin(), jp_triple_vector.end());
    std::cout << "jc nnz : " << Jc.nonZeros() << std::endl;
    std::cout << "jp nnz : " << Jp.nonZeros() << std::endl;
    jc_triple_vector.clear();
    jp_triple_vector.clear();
    Eigen::VectorXd delta_c, delta_p;
    SolveNormalFormula(Jc, Jp, r, mu, delta_c, delta_p);
    if (delta_c.norm() < epsilon * camera_variable.norm() && delta_p.norm() < epsilon * point_variable.norm()) {
      break;
    }
    auto camera_variable_old = camera_variable;
    auto point_variable_old = point_variable;
    double old_residuals_norm = r.norm();

    camera_variable +=  delta_c;
    point_variable += delta_p;


    for (auto &f : evaluate_functions) {
      f();
    }

    for (auto &f : jc_jp_functions) {
      f();
    }
    Jc.setFromTriplets(jc_triple_vector.begin(), jc_triple_vector.end());
    Jp.setFromTriplets(jp_triple_vector.begin(), jp_triple_vector.end());
    jc_triple_vector.clear();
    jp_triple_vector.clear();
    auto gc = -Jc.transpose() * r;
    auto gp = -Jp.transpose() * r;
    double rho = (old_residuals_norm - r.norm()) / (delta_c.dot(mu * delta_c + gc) + delta_p.dot(mu * delta_p + gp));

    if (rho > 0) {
      mu *= std::max(1.0 / 3, 1 - std::pow(2 * rho - 1, 3));
      v = 2;
    } else {
      camera_variable = camera_variable_old;
      point_variable = point_variable_old;
      mu *= v;
      v *= 2;
    }
    // Solver
    //std::cout << "camera_variable size : " << camera_variable.rows()
              //<< " delta_c : " << delta_c.rows() << std::endl;


    //std::cout << "point_variable size : " << point_variable.rows()
              //<< " delta_p : " << delta_p.rows() << std::endl;
  }

  
  // Write back

  for (auto [camera_index, idx] : camera_idx_to_Jc_idx) {
    double* ptr = camera_variable.data() + idx * 6;
    std::copy(ptr, ptr + 6, problem.cameras_[camera_index].data());
  }

  for (auto [point_idx, idx] : point_idx_to_Jp_idx) {
    double* ptr = point_variable.data() + idx * 3;
    std::copy(ptr, ptr + 3, problem.points_[point_idx].data());
  }
}
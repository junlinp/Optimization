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

template<class Jc, class Jp, class R, class DC, class DP>
void SolveNormalFormula(const Jc& jc, const Jp& jp, const R& r, DC& delta_c, DP& delta_p) {

  using SM = Eigen::SparseMatrix<double>;
  std::cout << "B" << std::endl;
  SM U = SM(jc.transpose()) * jc;
  SM W = SM(jc.transpose()) * jp;
  SM V = SM(jp.transpose()) * jp;
  
  std::cout << "Jc : " << jc.rows() << " , " << jc.cols() << std::endl;
  std::cout << "Jp : " << jp.rows() << " , " << jp.cols() << std::endl;
  std::cout << "V : " << V.rows() << " , " << V.cols() << std::endl;
  std::cout << "C" << std::endl;

  auto llt_solve = Eigen::SimplicialLLT<SM>();
  llt_solve.compute(V);

  std::cout << "V nnz : " << V.nonZeros() << std::endl;

  SM jpt = jp.transpose();
  std::cout << "jpt nnz : " << jpt.nonZeros() << std::endl;

  Eigen::VectorXd jpT_r = jp.transpose() * r;

  std::cout << "jp_T * r norm : " << jpT_r.norm() << std::endl;


  Eigen::MatrixXd g = -jc.transpose() * r + W * llt_solve.solve(jpT_r);

  std::cout << "a" << std::endl;

  Eigen::MatrixXd A = U - W * llt_solve.solve(Eigen::MatrixXd(W.transpose()));

  std::cout << "delta_c" << std::endl;
  delta_c = A.ldlt().solve(g);

  std::cout << "delta_p" << std::endl;
  std::cout << "r norm : " << r.norm() << std::endl;
  std::cout << "WT * delta_c norm : " << (W.transpose() * delta_c).norm() << std::endl;
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

  Eigen::VectorXd delta_c, delta_p;
  for (int k = 0; k < Jp.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(Jp, k); it; ++it) {
      it.value();
      it.row();   // row index
      it.col();   // col index (here it is equal to k)
      it.index(); // inner index, here it is equal to it.row()

      //std::cout << "Jp[" << it.row() << "," << it.col() << "] : " << it.value() << std::endl;
    }
  }
  SolveNormalFormula(Jc, Jp, r, delta_c, delta_p);
  // Solver
  std::cout << "camera_variable size : " << camera_variable.rows() << " delta_c : " << delta_c.rows() << std::endl;
  double alpha = 0.001;
  camera_variable += alpha * delta_c;

  std::cout << "point_variable size : " << point_variable.rows() << " delta_p : " << delta_p.rows() << std::endl;
  point_variable += alpha * delta_p;

  
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
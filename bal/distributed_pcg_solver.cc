#include "distributed_pcg_solver.h"
#include "Eigen/Sparse"
#include "cost_function_auto.h"
#include "problem.h"
#include <Eigen/src/IterativeLinearSolvers/ConjugateGradient.h>
#include <Eigen/src/SparseCholesky/SimplicialCholesky.h>
#include <ceres/cost_function.h>

#include <map>
#include <set>
#include <utility>

void DistributedPCGSolver::Solve(Problem &problem) {
  using SM = Eigen::SparseMatrix<double>;

  std::map<size_t, size_t> camera_idx_to_Jc_idx;
  std::map<size_t, size_t> point_idx_to_Jp_idx;

  using Triple = Eigen::Triplet<double>;
  std::vector<Triple> jc_triple_vector;
  std::vector<Triple> jp_triple_vector;

  size_t row_count = 0;

  Eigen::VectorXd r(problem.observations_.size() * 2);

  for (const auto &[edge, observation] : problem.observations_) {
    size_t camera_idx = edge.first;
    size_t point_idx = edge.first;
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
    const double *camera_intrinsics = camera_parameters.data() + 6;
    ceres::CostFunction *cost_function_ptr =
        ProjectFunction::CreateCostFunction(
            camera_intrinsics[0], camera_intrinsics[1], camera_intrinsics[2],
            observation.u(), observation.v());
    std::vector<double *> jacobians;
    double jacobian_camera[2 * ProjectFunction::CAMERA_PARAMETER_SIZE];
    double jacobian_point[2 * ProjectFunction::POINT_PARAMETER_SIZE];
    jacobians.push_back(jacobian_camera);
    jacobians.push_back(jacobian_point);

    std::vector<const double *> parameters = {camera_parameters.data(),
                                              point.data()};
    double res[2];
    cost_function_ptr->Evaluate(parameters.data(), res, jacobians.data());

    size_t jc_offset = camera_idx_to_Jc_idx.at(camera_idx) *
                       ProjectFunction::CAMERA_PARAMETER_SIZE;
    size_t jp_offset = point_idx_to_Jp_idx.at(point_idx) *
                       ProjectFunction::POINT_PARAMETER_SIZE;

    for (int k = 0; k < 2; k++) {
      for (int i = 0; i < ProjectFunction::CAMERA_PARAMETER_SIZE; i++) {
        Triple t{
            2 * row_count + k, jc_offset + i,
            jacobian_camera[k * ProjectFunction::CAMERA_PARAMETER_SIZE + i]};
        jc_triple_vector.push_back(t);
      }

      for (int i = 0; i < ProjectFunction::POINT_PARAMETER_SIZE; i++) {
        Triple t{2 * row_count + k, jp_offset + i,
                 jacobian_point[k * ProjectFunction::POINT_PARAMETER_SIZE + i]};
        jp_triple_vector.push_back(t);
      }
    }
    r(row_count * 2) = res[0];
    r(row_count * 2 + 1) = res[1];
    row_count++;
    delete cost_function_ptr;
  }

  size_t rows = problem.observations_.size() * 2;
  size_t col_of_jc = camera_idx_to_Jc_idx.size() * ProjectFunction::CAMERA_PARAMETER_SIZE;
  size_t col_of_jp = point_idx_to_Jp_idx.size() * ProjectFunction::POINT_PARAMETER_SIZE;
  SM Jc(rows, col_of_jc);
  SM Jp(rows, col_of_jp);

  std::cout << "jc : " << jc_triple_vector.size() << std::endl;
  std::cout << "jp : " << jp_triple_vector.size() << std::endl;

  Jc.setFromTriplets(jc_triple_vector.begin(), jc_triple_vector.end());
  Jp.setFromTriplets(jp_triple_vector.begin(), jp_triple_vector.end());

  // Build jacobian

  // Solver
  SM B = SM(Jc.transpose()) * Jc;
  SM E = SM(Jc.transpose()) * Jp;
  SM C = SM(Jp.transpose()) * Jp;
  SM ET(E.transpose());

  //
  //  [B   E]  [xc]  =  -[Jc^T]
  //  [E^T C]  [xp]      [Jp^T] r
  //
  //  (B - E * C^(-1) * E^T) * delta_c = -[Jc^T * E * C^(-1) * Jp^T] * r
  //  
  Eigen::SimplicialLDLT<SM> ldlt_solver;
  ldlt_solver.compute(C);
  Eigen::MatrixXd solution = ldlt_solver.solve(ET);
  Eigen::MatrixXd A = E * solution;

  Eigen::MatrixXd b = -(SM(Jc.transpose()) * E * ldlt_solver.solve(SM(Jp.transpose())));
  Eigen::ConjugateGradient<decltype(A)> cg_solver;
  //cg_solver.compute(A);
  //Eigen::VectorXd delta_x = cg_solver.solve(b);
  //
  // fine
}
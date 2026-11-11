#include "jordan_conic.h"
#include <iomanip>
#include <iostream>
#include <vector>

// Internal benchmark interchange: m n block_count, blocks (N/Q/S size),
// followed by row-major A, b, c. Results contain status, iterations,
// residuals, gap, and the vectors x, y, s on separate lines.
int main() {
  int m, n, count;
  if (!(std::cin >> m >> n >> count) || m < 0 || n <= 0 || count <= 0 || n > 2000 || m > 2000) return 2;
  std::vector<jordan::ConeBlock> blocks;
  for (int i=0;i<count;++i) {
    char type; int size;
    if (!(std::cin >> type >> size)) return 2;
    if (type == 'N') blocks.push_back({jordan::ConeType::Nonnegative,size});
    else if (type == 'Q') blocks.push_back({jordan::ConeType::Lorentz,size});
    else if (type == 'S') blocks.push_back({jordan::ConeType::Semidefinite,size});
    else return 2;
  }
  if (jordan::Dimension(blocks) != n) return 2;
  Eigen::MatrixXd A(m,n); Eigen::VectorXd b(m), c(n);
  for (int i=0;i<m;++i) for (int j=0;j<n;++j) if (!(std::cin >> A(i,j))) return 2;
  for (int i=0;i<m;++i) if (!(std::cin >> b(i))) return 2;
  for (int i=0;i<n;++i) if (!(std::cin >> c(i))) return 2;
  auto r=jordan::Solve(A,b,c,blocks);
  std::cout << std::setprecision(17) << static_cast<int>(r.status) << ' ' << r.iterations << ' '
            << r.primal_residual << ' ' << r.dual_residual << ' ' << r.gap << '\n';
  std::cout << r.x.transpose() << '\n' << r.y.transpose() << '\n' << r.s.transpose() << '\n';
}

#include "harris_corners.h"

#include <algorithm>
#include <map>

#include <Eigen/Dense>

namespace vio {
namespace {

// Standard 3x3 Sobel kernels.
double SobelX(const GrayImage& img, int r, int c) {
  return -1.0 * img.at(r - 1, c - 1) + 1.0 * img.at(r - 1, c + 1) - 2.0 * img.at(r, c - 1) +
        2.0 * img.at(r, c + 1) - 1.0 * img.at(r + 1, c - 1) + 1.0 * img.at(r + 1, c + 1);
}

double SobelY(const GrayImage& img, int r, int c) {
  return -1.0 * img.at(r - 1, c - 1) - 2.0 * img.at(r - 1, c) - 1.0 * img.at(r - 1, c + 1) +
        1.0 * img.at(r + 1, c - 1) + 2.0 * img.at(r + 1, c) + 1.0 * img.at(r + 1, c + 1);
}

}  // namespace

std::vector<Corner> DetectHarrisCorners(const GrayImage& image, const HarrisOptions& options) {
  const int rows = image.rows;
  const int cols = image.cols;
  const int border = options.window_radius + 1;  // Sobel's own 1px + box-sum window
  if (rows <= 2 * border || cols <= 2 * border) return {};

  Eigen::MatrixXd Ixx = Eigen::MatrixXd::Zero(rows, cols);
  Eigen::MatrixXd Iyy = Eigen::MatrixXd::Zero(rows, cols);
  Eigen::MatrixXd Ixy = Eigen::MatrixXd::Zero(rows, cols);
  for (int r = 1; r < rows - 1; ++r) {
    for (int c = 1; c < cols - 1; ++c) {
      const double gx = SobelX(image, r, c);
      const double gy = SobelY(image, r, c);
      Ixx(r, c) = gx * gx;
      Iyy(r, c) = gy * gy;
      Ixy(r, c) = gx * gy;
    }
  }

  Eigen::MatrixXd response = Eigen::MatrixXd::Zero(rows, cols);
  double max_response = 0.0;
  const int w = options.window_radius;
  for (int r = border; r < rows - border; ++r) {
    for (int c = border; c < cols - border; ++c) {
      double sxx = 0, syy = 0, sxy = 0;
      for (int dr = -w; dr <= w; ++dr) {
        for (int dc = -w; dc <= w; ++dc) {
          sxx += Ixx(r + dr, c + dc);
          syy += Iyy(r + dr, c + dc);
          sxy += Ixy(r + dr, c + dc);
        }
      }
      const double det = sxx * syy - sxy * sxy;
      const double trace = sxx + syy;
      const double r_score = det - options.k * trace * trace;
      response(r, c) = r_score;
      max_response = std::max(max_response, r_score);
    }
  }

  const double threshold = options.response_threshold_ratio * max_response;

  std::vector<Corner> survivors;
  for (int r = border; r < rows - border; ++r) {
    for (int c = border; c < cols - border; ++c) {
      const double score = response(r, c);
      if (score <= threshold || score <= 0.0) continue;

      bool is_local_max = true;
      for (int dr = -options.nms_radius; dr <= options.nms_radius && is_local_max; ++dr) {
        for (int dc = -options.nms_radius; dc <= options.nms_radius; ++dc) {
          if (dr == 0 && dc == 0) continue;
          const int nr = r + dr, nc = c + dc;
          if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
          if (response(nr, nc) > score) {
            is_local_max = false;
            break;
          }
        }
      }
      if (is_local_max) survivors.push_back({static_cast<double>(c), static_cast<double>(r), score});
    }
  }

  // Grid-bucket: keep only the top max_per_cell survivors in each
  // cell_size x cell_size cell, for spatial distribution.
  std::map<std::pair<int, int>, std::vector<Corner>> cells;
  for (const Corner& corner : survivors) {
    const int cell_r = static_cast<int>(corner.y) / options.cell_size;
    const int cell_c = static_cast<int>(corner.x) / options.cell_size;
    cells[{cell_r, cell_c}].push_back(corner);
  }

  std::vector<Corner> bucketed;
  for (auto& [key, corners] : cells) {
    std::sort(corners.begin(), corners.end(),
             [](const Corner& a, const Corner& b) { return a.score > b.score; });
    const size_t keep = std::min(corners.size(), static_cast<size_t>(options.max_per_cell));
    bucketed.insert(bucketed.end(), corners.begin(), corners.begin() + keep);
  }

  std::sort(bucketed.begin(), bucketed.end(),
           [](const Corner& a, const Corner& b) { return a.score > b.score; });
  const size_t keep_total =
      std::min(bucketed.size(), static_cast<size_t>(options.max_total_features));
  bucketed.resize(keep_total);
  return bucketed;
}

}  // namespace vio

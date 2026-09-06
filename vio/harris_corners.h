#ifndef VIO_HARRIS_CORNERS_H_
#define VIO_HARRIS_CORNERS_H_
#include <vector>

#include "image_io.h"

namespace vio {

struct Corner {
  double x = 0, y = 0, score = 0;
};

struct HarrisOptions {
  int window_radius = 2;  // structure-tensor box-sum window: (2r+1)x(2r+1) = 5x5
  double k = 0.04;
  double response_threshold_ratio = 0.01;  // keep pixels with R > ratio * max(R)
  int nms_radius = 3;
  int cell_size = 40;
  int max_per_cell = 5;
  int max_total_features = 200;
};

std::vector<Corner> DetectHarrisCorners(const GrayImage& image, const HarrisOptions& options);

}  // namespace vio
#endif  // VIO_HARRIS_CORNERS_H_

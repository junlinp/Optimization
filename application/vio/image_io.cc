#include "image_io.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "stb_image.h"

namespace vio {

double GrayImage::Bilinear(double r, double c) const {
  const int r0 = static_cast<int>(std::floor(r));
  const int c0 = static_cast<int>(std::floor(c));
  const int r1 = r0 + 1;
  const int c1 = c0 + 1;
  if (!InBounds(r0, c0) || !InBounds(r0, c1) || !InBounds(r1, c0) || !InBounds(r1, c1)) {
    return 0.0;
  }
  const double fr = r - r0;
  const double fc = c - c0;
  const double top = at(r0, c0) * (1.0 - fc) + at(r0, c1) * fc;
  const double bottom = at(r1, c0) * (1.0 - fc) + at(r1, c1) * fc;
  return top * (1.0 - fr) + bottom * fr;
}

GrayImage LoadGrayscalePng(const std::string& path) {
  int w = 0, h = 0, channels_in_file = 0;
  unsigned char* data = stbi_load(path.c_str(), &w, &h, &channels_in_file, 1);
  if (!data) {
    throw std::runtime_error("image_io: failed to load image: " + path);
  }

  GrayImage image;
  image.rows = h;
  image.cols = w;
  image.pixels.assign(data, data + static_cast<size_t>(w) * static_cast<size_t>(h));
  stbi_image_free(data);
  return image;
}

}  // namespace vio

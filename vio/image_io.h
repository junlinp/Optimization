#ifndef VIO_IMAGE_IO_H_
#define VIO_IMAGE_IO_H_
#include <cstdint>
#include <string>
#include <vector>

namespace vio {

struct GrayImage {
  int rows = 0, cols = 0;
  std::vector<uint8_t> pixels;  // row-major, size rows*cols

  uint8_t at(int r, int c) const { return pixels[static_cast<size_t>(r) * cols + c]; }
  bool InBounds(int r, int c) const { return r >= 0 && r < rows && c >= 0 && c < cols; }

  // Bilinear sample; returns 0.0 for out-of-bounds (r,c) -- used by the
  // rectify remap and patch matching, both of which read at fractional and
  // possibly-out-of-range coordinates.
  double Bilinear(double r, double c) const;
};

// Loads via stbi_load(path, &w, &h, &c, 1) (forces single-channel grayscale;
// EuRoC PNGs are already grayscale). Throws std::runtime_error on failure.
GrayImage LoadGrayscalePng(const std::string& path);

}  // namespace vio
#endif  // VIO_IMAGE_IO_H_

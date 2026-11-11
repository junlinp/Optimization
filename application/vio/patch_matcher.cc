#include "patch_matcher.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vio {
namespace {

double PatchSSD(const GrayImage& a, double ax, double ay, const GrayImage& b, double bx, double by,
                int radius) {
  double sum = 0.0;
  for (int dr = -radius; dr <= radius; ++dr) {
    for (int dc = -radius; dc <= radius; ++dc) {
      const double va = a.Bilinear(ay + dr, ax + dc);
      const double vb = b.Bilinear(by + dr, bx + dc);
      const double diff = va - vb;
      sum += diff * diff;
    }
  }
  return sum;
}

bool PatchWindowInBounds(const GrayImage& img, double x, double y, int radius) {
  return x - radius >= 0 && x + radius <= img.cols - 1 && y - radius >= 0 &&
        y + radius <= img.rows - 1;
}

double PatchVariance(const GrayImage& img, double x, double y, int radius) {
  double sum = 0.0;
  const int n = (2 * radius + 1) * (2 * radius + 1);
  for (int dr = -radius; dr <= radius; ++dr) {
    for (int dc = -radius; dc <= radius; ++dc) {
      sum += img.Bilinear(y + dr, x + dc);
    }
  }
  const double mean = sum / n;
  double sq_sum = 0.0;
  for (int dr = -radius; dr <= radius; ++dr) {
    for (int dc = -radius; dc <= radius; ++dc) {
      const double diff = img.Bilinear(y + dr, x + dc) - mean;
      sq_sum += diff * diff;
    }
  }
  return sq_sum / n;
}

}  // namespace

bool MatchStereoPatch(const GrayImage& left, const GrayImage& right, double u_left, double v_left,
                      const PatchMatchOptions& options, double* disparity, double* score) {
  if (!PatchWindowInBounds(left, u_left, v_left, options.patch_radius)) return false;
  if (PatchVariance(left, u_left, v_left, options.patch_radius) < options.min_query_variance) {
    return false;
  }

  double best_ssd = std::numeric_limits<double>::max();
  int best_d = -1;
  for (int d = options.stereo_min_disparity; d <= options.stereo_max_disparity; ++d) {
    const double u_right = u_left - d;
    if (!PatchWindowInBounds(right, u_right, v_left, options.patch_radius)) continue;
    const double ssd = PatchSSD(left, u_left, v_left, right, u_right, v_left, options.patch_radius);
    if (ssd < best_ssd) {
      best_ssd = ssd;
      best_d = d;
    }
  }
  if (best_d < 0) return false;

  const int patch_pixels = (2 * options.patch_radius + 1) * (2 * options.patch_radius + 1);
  if (best_ssd / patch_pixels > options.max_ssd_per_pixel) return false;

  *disparity = best_d;
  *score = best_ssd;
  return true;
}

bool MatchTemporalPatch(const GrayImage& prev, const GrayImage& curr, double u_prev, double v_prev,
                        const PatchMatchOptions& options, double* u_curr, double* v_curr,
                        double* score) {
  return MatchTemporalPatchNear(prev, curr, u_prev, v_prev, u_prev, v_prev,
                                options.temporal_search_radius, options, u_curr, v_curr, score);
}

bool MatchTemporalPatchNear(const GrayImage& prev, const GrayImage& curr, double u_prev,
                            double v_prev, double u_guess, double v_guess, int search_radius,
                            const PatchMatchOptions& options, double* u_curr, double* v_curr,
                            double* score) {
  if (!PatchWindowInBounds(prev, u_prev, v_prev, options.patch_radius)) return false;
  if (PatchVariance(prev, u_prev, v_prev, options.patch_radius) < options.min_query_variance) {
    return false;
  }

  double best_ssd = std::numeric_limits<double>::max();
  int best_du = 0, best_dv = 0;
  bool found = false;
  for (int dv = -search_radius; dv <= search_radius; ++dv) {
    for (int du = -search_radius; du <= search_radius; ++du) {
      const double uc = u_guess + du;
      const double vc = v_guess + dv;
      if (!PatchWindowInBounds(curr, uc, vc, options.patch_radius)) continue;
      const double ssd = PatchSSD(prev, u_prev, v_prev, curr, uc, vc, options.patch_radius);
      if (ssd < best_ssd) {
        best_ssd = ssd;
        best_du = du;
        best_dv = dv;
        found = true;
      }
    }
  }
  if (!found) return false;

  const int patch_pixels = (2 * options.patch_radius + 1) * (2 * options.patch_radius + 1);
  if (best_ssd / patch_pixels > options.max_ssd_per_pixel) return false;

  *u_curr = u_guess + best_du;
  *v_curr = v_guess + best_dv;
  *score = best_ssd;
  if (options.temporal_subpixel) {
    // Vertex of the parabola through (-1, a), (0, b), (1, c): (a - c) / (2(a - 2b + c)).
    auto refine = [&](double du, double dv) {
      const double uc = *u_curr + du, vc = *v_curr + dv;
      if (!PatchWindowInBounds(curr, uc, vc, options.patch_radius)) return best_ssd;
      return PatchSSD(prev, u_prev, v_prev, curr, uc, vc, options.patch_radius);
    };
    auto vertex = [](double a, double b, double c) {
      const double denom = a - 2 * b + c;
      return denom > 0 ? std::max(-0.5, std::min(0.5, 0.5 * (a - c) / denom)) : 0.0;
    };
    const double offset_u = vertex(refine(-1, 0), best_ssd, refine(1, 0));
    const double offset_v = vertex(refine(0, -1), best_ssd, refine(0, 1));
    *u_curr += offset_u;
    *v_curr += offset_v;
  }
  return true;
}

}  // namespace vio

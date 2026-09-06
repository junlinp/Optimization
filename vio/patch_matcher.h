#ifndef VIO_PATCH_MATCHER_H_
#define VIO_PATCH_MATCHER_H_
#include "image_io.h"

namespace vio {

struct PatchMatchOptions {
  int patch_radius = 3;  // (2r+1)x(2r+1) = 7x7 patch
  int stereo_min_disparity = 1;
  int stereo_max_disparity = 128;
  int temporal_search_radius = 20;   // pixels, half-width of 2D search window
  double max_ssd_per_pixel = 4000.0; // reject if best SSD/pixel-count exceeds this
  // Reject up front if the query patch itself is this flat (intensity
  // variance below the threshold): every candidate would tie for "best"
  // match, making any single answer meaningless rather than a genuine
  // correspondence.
  double min_query_variance = 4.0;
};

// 1D search along the same row within [stereo_min_disparity, stereo_max_disparity].
bool MatchStereoPatch(const GrayImage& left, const GrayImage& right, double u_left, double v_left,
                      const PatchMatchOptions& options, double* disparity, double* score);

// 2D search in a (2*temporal_search_radius+1) window around (u_prev, v_prev).
bool MatchTemporalPatch(const GrayImage& prev, const GrayImage& curr, double u_prev, double v_prev,
                        const PatchMatchOptions& options, double* u_curr, double* v_curr,
                        double* score);

}  // namespace vio
#endif  // VIO_PATCH_MATCHER_H_

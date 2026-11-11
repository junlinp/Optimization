#ifndef SUPER_RESOLUTION_TV_SR_ADMM_H_
#define SUPER_RESOLUTION_TV_SR_ADMM_H_

#include <Eigen/Dense>

struct TvSrOptions {
    int scale = 4;
    double lambda = 0.004;
    double rho = 1.0;
    double blur_sigma = 2.0;
    int max_iters = 25;
    int cg_iters = 8;
    bool verbose = true;
};

Eigen::VectorXd GaussianKernel(double sigma);

Eigen::MatrixXd BlurDownsample(const Eigen::MatrixXd& hr, int scale,
                               double blur_sigma);

Eigen::MatrixXd BilinearResize(const Eigen::MatrixXd& src, int out_rows,
                               int out_cols);

Eigen::MatrixXd BicubicResize(const Eigen::MatrixXd& src, int out_rows,
                              int out_cols);

double Psnr(const Eigen::MatrixXd& ref, const Eigen::MatrixXd& rec);

// Reconstruct an HR image from a single LR observation with anisotropic TV-ADMM.
// hr_rows and hr_cols must be LR size times scale.
Eigen::MatrixXd SuperResolveAdmm(const Eigen::MatrixXd& lr,
                                 const TvSrOptions& options, int hr_rows,
                                 int hr_cols);

#endif  // SUPER_RESOLUTION_TV_SR_ADMM_H_

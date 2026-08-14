#include "gtest/gtest.h"

#include "tv_sr_admm.h"

TEST(TvSrAdmm, RecoversPiecewiseConstant) {
    Eigen::MatrixXd hr = Eigen::MatrixXd::Constant(32, 32, 0.2);
    hr.block(8, 8, 16, 16).setConstant(0.8);

    TvSrOptions opt;
    opt.scale = 2;
    opt.blur_sigma = 1.0;
    opt.lambda = 0.002;
    opt.rho = 1.0;
    opt.max_iters = 25;
    opt.cg_iters = 12;
    opt.verbose = false;

    const Eigen::MatrixXd lr = BlurDownsample(hr, opt.scale, opt.blur_sigma);
    const Eigen::MatrixXd sr = SuperResolveAdmm(lr, opt, 32, 32);
    const Eigen::MatrixXd bicubic = BicubicResize(lr, 32, 32);

    EXPECT_EQ(sr.rows(), 32);
    EXPECT_EQ(sr.cols(), 32);
    EXPECT_GE(sr.minCoeff(), -1e-6);
    EXPECT_LE(sr.maxCoeff(), 1.0 + 1e-6);
    EXPECT_GT(Psnr(hr, sr), 20.0);
    EXPECT_GT(Psnr(hr, sr), Psnr(hr, bicubic));
}

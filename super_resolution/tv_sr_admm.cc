#include "tv_sr_admm.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int Clampi(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }

double SampleReplicate(const Eigen::MatrixXd& src, int r, int c) {
    return src(Clampi(r, 0, static_cast<int>(src.rows()) - 1),
               Clampi(c, 0, static_cast<int>(src.cols()) - 1));
}

Eigen::MatrixXd SeparableConv(const Eigen::MatrixXd& x, const Eigen::VectorXd& k) {
    const int radius = static_cast<int>(k.size() / 2);
    const int h = static_cast<int>(x.rows());
    const int w = static_cast<int>(x.cols());
    Eigen::MatrixXd tmp(h, w);
    Eigen::MatrixXd out(h, w);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            double acc = 0.0;
            for (int t = -radius; t <= radius; ++t) {
                acc += k(t + radius) * SampleReplicate(x, i, j + t);
            }
            tmp(i, j) = acc;
        }
    }
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            double acc = 0.0;
            for (int t = -radius; t <= radius; ++t) {
                acc += k(t + radius) * SampleReplicate(tmp, i + t, j);
            }
            out(i, j) = acc;
        }
    }
    return out;
}

Eigen::MatrixXd DownsampleMean(const Eigen::MatrixXd& x, int scale) {
    const int h = static_cast<int>(x.rows()) / scale;
    const int w = static_cast<int>(x.cols()) / scale;
    Eigen::MatrixXd y(h, w);
    const double inv = 1.0 / (scale * scale);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            double acc = 0.0;
            for (int di = 0; di < scale; ++di) {
                for (int dj = 0; dj < scale; ++dj) {
                    acc += x(i * scale + di, j * scale + dj);
                }
            }
            y(i, j) = acc * inv;
        }
    }
    return y;
}

Eigen::MatrixXd UpsampleMeanAdjoint(const Eigen::MatrixXd& y, int scale,
                                    int hr_rows, int hr_cols) {
    Eigen::MatrixXd x = Eigen::MatrixXd::Zero(hr_rows, hr_cols);
    const double inv = 1.0 / (scale * scale);
    const int h = static_cast<int>(y.rows());
    const int w = static_cast<int>(y.cols());
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            const double v = y(i, j) * inv;
            for (int di = 0; di < scale; ++di) {
                for (int dj = 0; dj < scale; ++dj) {
                    x(i * scale + di, j * scale + dj) = v;
                }
            }
        }
    }
    return x;
}

void Dh(const Eigen::MatrixXd& x, Eigen::MatrixXd* dx) {
    dx->resize(x.rows(), x.cols());
    const int w = static_cast<int>(x.cols());
    if (w == 1) {
        dx->setZero();
        return;
    }
    dx->leftCols(w - 1) = x.rightCols(w - 1) - x.leftCols(w - 1);
    dx->col(w - 1).setZero();
}

void Dv(const Eigen::MatrixXd& x, Eigen::MatrixXd* dx) {
    dx->resize(x.rows(), x.cols());
    const int h = static_cast<int>(x.rows());
    if (h == 1) {
        dx->setZero();
        return;
    }
    dx->topRows(h - 1) = x.bottomRows(h - 1) - x.topRows(h - 1);
    dx->row(h - 1).setZero();
}

void DhT(const Eigen::MatrixXd& z, Eigen::MatrixXd* y) {
    y->resize(z.rows(), z.cols());
    const int w = static_cast<int>(z.cols());
    y->col(0) = -z.col(0);
    for (int j = 1; j < w - 1; ++j) {
        y->col(j) = z.col(j - 1) - z.col(j);
    }
    if (w > 1) {
        y->col(w - 1) = z.col(w - 2);
    }
}

void DvT(const Eigen::MatrixXd& z, Eigen::MatrixXd* y) {
    y->resize(z.rows(), z.cols());
    const int h = static_cast<int>(z.rows());
    y->row(0) = -z.row(0);
    for (int i = 1; i < h - 1; ++i) {
        y->row(i) = z.row(i - 1) - z.row(i);
    }
    if (h > 1) {
        y->row(h - 1) = z.row(h - 2);
    }
}

Eigen::MatrixXd SoftThreshold(const Eigen::MatrixXd& v, double tau) {
    return v.unaryExpr([tau](double x) {
        const double ax = std::abs(x);
        return (ax > tau) ? std::copysign(ax - tau, x) : 0.0;
    });
}

double CubicWeight(double t) {
    const double a = -0.5;
    t = std::abs(t);
    if (t <= 1.0) {
        return ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0;
    }
    if (t < 2.0) {
        return (((a * t - 5.0 * a) * t + 8.0 * a) * t - 4.0 * a);
    }
    return 0.0;
}

}  // namespace

Eigen::VectorXd GaussianKernel(double sigma) {
    sigma = std::max(sigma, 0.3);
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
    Eigen::VectorXd k(2 * radius + 1);
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        const double v = std::exp(-0.5 * static_cast<double>(i * i) / (sigma * sigma));
        k(i + radius) = v;
        sum += v;
    }
    return k / sum;
}

Eigen::MatrixXd BlurDownsample(const Eigen::MatrixXd& hr, int scale,
                               double blur_sigma) {
    return DownsampleMean(SeparableConv(hr, GaussianKernel(blur_sigma)), scale);
}

Eigen::MatrixXd BilinearResize(const Eigen::MatrixXd& src, int out_rows,
                               int out_cols) {
    Eigen::MatrixXd dst(out_rows, out_cols);
    const double scale_r =
        static_cast<double>(src.rows()) / static_cast<double>(out_rows);
    const double scale_c =
        static_cast<double>(src.cols()) / static_cast<double>(out_cols);
    for (int i = 0; i < out_rows; ++i) {
        const double rf = (i + 0.5) * scale_r - 0.5;
        const int r0 = Clampi(static_cast<int>(std::floor(rf)), 0,
                              static_cast<int>(src.rows()) - 1);
        const int r1 = Clampi(r0 + 1, 0, static_cast<int>(src.rows()) - 1);
        const double dr = rf - r0;
        for (int j = 0; j < out_cols; ++j) {
            const double cf = (j + 0.5) * scale_c - 0.5;
            const int c0 = Clampi(static_cast<int>(std::floor(cf)), 0,
                                  static_cast<int>(src.cols()) - 1);
            const int c1 = Clampi(c0 + 1, 0, static_cast<int>(src.cols()) - 1);
            const double dc = cf - c0;
            const double v00 = src(r0, c0);
            const double v01 = src(r0, c1);
            const double v10 = src(r1, c0);
            const double v11 = src(r1, c1);
            dst(i, j) = (1.0 - dr) * ((1.0 - dc) * v00 + dc * v01) +
                        dr * ((1.0 - dc) * v10 + dc * v11);
        }
    }
    return dst;
}

Eigen::MatrixXd BicubicResize(const Eigen::MatrixXd& src, int out_rows,
                              int out_cols) {
    Eigen::MatrixXd dst(out_rows, out_cols);
    const double scale_r =
        static_cast<double>(src.rows()) / static_cast<double>(out_rows);
    const double scale_c =
        static_cast<double>(src.cols()) / static_cast<double>(out_cols);
    for (int i = 0; i < out_rows; ++i) {
        const double rf = (i + 0.5) * scale_r - 0.5;
        const int r_base = static_cast<int>(std::floor(rf));
        for (int j = 0; j < out_cols; ++j) {
            const double cf = (j + 0.5) * scale_c - 0.5;
            const int c_base = static_cast<int>(std::floor(cf));
            double acc = 0.0;
            for (int n = -1; n <= 2; ++n) {
                const double wy = CubicWeight(rf - (r_base + n));
                for (int m = -1; m <= 2; ++m) {
                    const double wx = CubicWeight(cf - (c_base + m));
                    acc += wy * wx * SampleReplicate(src, r_base + n, c_base + m);
                }
            }
            dst(i, j) = acc;
        }
    }
    return dst;
}

double Psnr(const Eigen::MatrixXd& ref, const Eigen::MatrixXd& rec) {
    const double mse = (ref - rec).squaredNorm() / static_cast<double>(ref.size());
    if (mse <= 1e-12) {
        return 99.0;
    }
    return 10.0 * std::log10(1.0 / mse);
}

Eigen::MatrixXd SuperResolveAdmm(const Eigen::MatrixXd& lr,
                                 const TvSrOptions& options, int hr_rows,
                                 int hr_cols) {
    const int scale = options.scale;
    if (hr_rows % scale != 0 || hr_cols % scale != 0) {
        throw std::runtime_error("HR size must be divisible by the scale factor");
    }
    if (lr.rows() * scale != hr_rows || lr.cols() * scale != hr_cols) {
        throw std::runtime_error("LR size must equal HR size / scale");
    }

    const Eigen::VectorXd kernel = GaussianKernel(options.blur_sigma);
    const double rho = options.rho;
    const double tau = options.lambda / rho;

    auto apply_k = [&](const Eigen::MatrixXd& x) -> Eigen::MatrixXd {
        return DownsampleMean(SeparableConv(x, kernel), scale);
    };
    auto apply_kt = [&](const Eigen::MatrixXd& y) -> Eigen::MatrixXd {
        return SeparableConv(UpsampleMeanAdjoint(y, scale, hr_rows, hr_cols), kernel);
    };
    auto apply_system = [&](const Eigen::MatrixXd& x) -> Eigen::MatrixXd {
        Eigen::MatrixXd dh, dv, dth, dtv;
        Dh(x, &dh);
        Dv(x, &dv);
        DhT(dh, &dth);
        DvT(dv, &dtv);
        return apply_kt(apply_k(x)) + rho * (dth + dtv);
    };

    Eigen::MatrixXd x = BicubicResize(lr, hr_rows, hr_cols);
    Eigen::MatrixXd zh = Eigen::MatrixXd::Zero(hr_rows, hr_cols);
    Eigen::MatrixXd zv = Eigen::MatrixXd::Zero(hr_rows, hr_cols);
    Eigen::MatrixXd uh = Eigen::MatrixXd::Zero(hr_rows, hr_cols);
    Eigen::MatrixXd uv = Eigen::MatrixXd::Zero(hr_rows, hr_cols);
    const Eigen::MatrixXd kty = apply_kt(lr);

    Eigen::MatrixXd dh, dv, dth, dtv;
    for (int iter = 0; iter < options.max_iters; ++iter) {
        DhT(zh - uh, &dth);
        DvT(zv - uv, &dtv);
        Eigen::MatrixXd rhs = kty + rho * (dth + dtv);

        Eigen::MatrixXd r = rhs - apply_system(x);
        Eigen::MatrixXd p = r;
        double rsold = r.squaredNorm();
        for (int cg = 0; cg < options.cg_iters && rsold > 1e-12; ++cg) {
            const Eigen::MatrixXd ap = apply_system(p);
            const double denom = p.cwiseProduct(ap).sum();
            if (std::abs(denom) < 1e-18) {
                break;
            }
            const double alpha = rsold / denom;
            x += alpha * p;
            r -= alpha * ap;
            const double rsnew = r.squaredNorm();
            p = r + (rsnew / rsold) * p;
            rsold = rsnew;
        }

        Dh(x, &dh);
        Dv(x, &dv);
        zh = SoftThreshold(dh + uh, tau);
        zv = SoftThreshold(dv + uv, tau);
        uh += dh - zh;
        uv += dv - zv;

        if (options.verbose && (iter == 0 || (iter + 1) % 5 == 0 ||
                                iter + 1 == options.max_iters)) {
            const double data = 0.5 * (apply_k(x) - lr).squaredNorm();
            const double tv = dh.cwiseAbs().sum() + dv.cwiseAbs().sum();
            std::cout << "ADMM iter " << (iter + 1) << "/" << options.max_iters
                      << " data=" << data << " tv=" << tv
                      << " obj=" << data + options.lambda * tv << std::endl;
        }
    }

    return x.cwiseMax(0.0).cwiseMin(1.0);
}

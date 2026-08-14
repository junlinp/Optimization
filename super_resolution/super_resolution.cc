#include "tv_sr_admm.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "stb_image.h"
#include "stb_image_write.h"

namespace {

constexpr int kUhdWidth = 3840;
constexpr int kUhdHeight = 2160;
constexpr int kScale = 4;

std::string FindLena(const std::string& explicit_path) {
    if (!explicit_path.empty()) {
        std::ifstream in(explicit_path);
        if (in.good()) {
            return explicit_path;
        }
    }
    const std::vector<std::string> candidates = {
        "super_resolution/testdata/lena.png",
        "testdata/lena.png",
        std::string(std::getenv("TEST_SRCDIR") ? std::getenv("TEST_SRCDIR") : ".") +
            "/Optimization/super_resolution/testdata/lena.png",
        std::string(std::getenv("TEST_SRCDIR") ? std::getenv("TEST_SRCDIR") : ".") +
            "/_main/super_resolution/testdata/lena.png",
    };
    for (const auto& path : candidates) {
        std::ifstream in(path);
        if (in.good()) {
            return path;
        }
    }
    return explicit_path.empty() ? candidates.front() : explicit_path;
}

struct RgbImage {
    int rows = 0;
    int cols = 0;
    Eigen::MatrixXd r;
    Eigen::MatrixXd g;
    Eigen::MatrixXd b;
};

RgbImage LoadRgb(const std::string& path) {
    int w = 0, h = 0, c = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!data) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    RgbImage img;
    img.rows = h;
    img.cols = w;
    img.r.resize(h, w);
    img.g.resize(h, w);
    img.b.resize(h, w);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            const int idx = (i * w + j) * 3;
            img.r(i, j) = data[idx + 0] / 255.0;
            img.g(i, j) = data[idx + 1] / 255.0;
            img.b(i, j) = data[idx + 2] / 255.0;
        }
    }
    stbi_image_free(data);
    return img;
}

void SaveRgb(const std::string& path, const RgbImage& img) {
    const int h = img.rows;
    const int w = img.cols;
    std::vector<unsigned char> bytes(static_cast<size_t>(h * w * 3));
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            const int idx = (i * w + j) * 3;
            auto to_u8 = [](double v) {
                v = std::max(0.0, std::min(1.0, v));
                return static_cast<unsigned char>(std::lround(v * 255.0));
            };
            bytes[idx + 0] = to_u8(img.r(i, j));
            bytes[idx + 1] = to_u8(img.g(i, j));
            bytes[idx + 2] = to_u8(img.b(i, j));
        }
    }
    if (!stbi_write_png(path.c_str(), w, h, 3, bytes.data(), w * 3)) {
        throw std::runtime_error("Failed to write image: " + path);
    }
}

Eigen::MatrixXd ToY(const RgbImage& img) {
    return 0.299 * img.r + 0.587 * img.g + 0.114 * img.b;
}

void RgbToYcbcr(const RgbImage& img, Eigen::MatrixXd* y, Eigen::MatrixXd* cb,
                Eigen::MatrixXd* cr) {
    *y = ToY(img);
    *cb = (-0.168736 * img.r - 0.331264 * img.g + 0.5 * img.b).array() + 0.5;
    *cr = (0.5 * img.r - 0.418688 * img.g - 0.081312 * img.b).array() + 0.5;
}

RgbImage YcbcrToRgb(const Eigen::MatrixXd& y, const Eigen::MatrixXd& cb,
                    const Eigen::MatrixXd& cr) {
    RgbImage img;
    img.rows = static_cast<int>(y.rows());
    img.cols = static_cast<int>(y.cols());
    const Eigen::MatrixXd cbp = cb.array() - 0.5;
    const Eigen::MatrixXd crp = cr.array() - 0.5;
    img.r = y + 1.402 * crp;
    img.g = y - 0.344136 * cbp - 0.714136 * crp;
    img.b = y + 1.772 * cbp;
    return img;
}

RgbImage ResizeRgb(const RgbImage& src, int out_rows, int out_cols, bool bicubic) {
    RgbImage dst;
    dst.rows = out_rows;
    dst.cols = out_cols;
    auto resize = [&](const Eigen::MatrixXd& ch) {
        return bicubic ? BicubicResize(ch, out_rows, out_cols)
                       : BilinearResize(ch, out_rows, out_cols);
    };
    dst.r = resize(src.r);
    dst.g = resize(src.g);
    dst.b = resize(src.b);
    return dst;
}

RgbImage MakeUhdCanvas(const RgbImage& lena) {
    const int side = kUhdHeight;
    RgbImage square = ResizeRgb(lena, side, side, /*bicubic=*/true);
    RgbImage canvas;
    canvas.rows = kUhdHeight;
    canvas.cols = kUhdWidth;
    canvas.r = Eigen::MatrixXd::Zero(kUhdHeight, kUhdWidth);
    canvas.g = Eigen::MatrixXd::Zero(kUhdHeight, kUhdWidth);
    canvas.b = Eigen::MatrixXd::Zero(kUhdHeight, kUhdWidth);
    const int x0 = (kUhdWidth - side) / 2;
    canvas.r.block(0, x0, side, side) = square.r;
    canvas.g.block(0, x0, side, side) = square.g;
    canvas.b.block(0, x0, side, side) = square.b;
    for (int j = 0; j < x0; ++j) {
        canvas.r.col(j) = square.r.col(0);
        canvas.g.col(j) = square.g.col(0);
        canvas.b.col(j) = square.b.col(0);
    }
    for (int j = x0 + side; j < kUhdWidth; ++j) {
        canvas.r.col(j) = square.r.col(side - 1);
        canvas.g.col(j) = square.g.col(side - 1);
        canvas.b.col(j) = square.b.col(side - 1);
    }
    return canvas;
}

RgbImage BlurDownsampleRgb(const RgbImage& hr, int scale, double sigma) {
    RgbImage lr;
    lr.r = BlurDownsample(hr.r, scale, sigma);
    lr.g = BlurDownsample(hr.g, scale, sigma);
    lr.b = BlurDownsample(hr.b, scale, sigma);
    lr.rows = static_cast<int>(lr.r.rows());
    lr.cols = static_cast<int>(lr.r.cols());
    return lr;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string input;
    std::string output = "lena_sr_4k.png";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--input" || arg == "-i") && i + 1 < argc) {
            input = argv[++i];
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            output = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Single-image TV super-resolution with ADMM.\n"
                << "Uses Lena to synthesize a 3840x2160 (4K) test, downsamples "
                   "4x, then reconstructs.\n\n"
                << "Usage: " << argv[0] << " [--input lena.png] [--output lena_sr_4k.png]\n";
            return 0;
        } else if (input.empty() && arg[0] != '-') {
            input = arg;
        }
    }

    try {
        const std::string lena_path = FindLena(input);
        std::cout << "Loading " << lena_path << std::endl;
        const RgbImage lena = LoadRgb(lena_path);
        std::cout << "Lena " << lena.cols << "x" << lena.rows << std::endl;

        const RgbImage hr = MakeUhdCanvas(lena);
        std::cout << "Synthetic 4K HR " << hr.cols << "x" << hr.rows << std::endl;

        TvSrOptions opt;
        opt.scale = kScale;
        opt.blur_sigma = 0.5 * kScale;
        opt.lambda = 0.001;
        opt.rho = 1.0;
        opt.max_iters = 20;
        opt.cg_iters = 8;
        opt.verbose = true;

        const RgbImage lr = BlurDownsampleRgb(hr, opt.scale, opt.blur_sigma);
        std::cout << "LR observation " << lr.cols << "x" << lr.rows << std::endl;

        Eigen::MatrixXd lr_y, lr_cb, lr_cr;
        RgbToYcbcr(lr, &lr_y, &lr_cb, &lr_cr);

        const auto t0 = std::chrono::steady_clock::now();
        const Eigen::MatrixXd sr_y =
            SuperResolveAdmm(lr_y, opt, hr.rows, hr.cols);
        const auto elapsed_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0)
                .count();

        const Eigen::MatrixXd sr_cb =
            BilinearResize(lr_cb, hr.rows, hr.cols);
        const Eigen::MatrixXd sr_cr =
            BilinearResize(lr_cr, hr.rows, hr.cols);
        const RgbImage sr = YcbcrToRgb(sr_y, sr_cb, sr_cr);
        const RgbImage bicubic = ResizeRgb(lr, hr.rows, hr.cols, /*bicubic=*/true);

        const double psnr_sr = Psnr(ToY(hr), sr_y);
        const double psnr_bicubic = Psnr(ToY(hr), ToY(bicubic));
        std::cout << "ADMM elapsed: " << elapsed_ms << " ms" << std::endl;
        std::cout << "Y PSNR bicubic: " << psnr_bicubic << " dB" << std::endl;
        std::cout << "Y PSNR ADMM:    " << psnr_sr << " dB" << std::endl;

        const std::string out_dir =
            std::getenv("BUILD_WORKING_DIRECTORY")
                ? std::string(std::getenv("BUILD_WORKING_DIRECTORY")) + "/"
                : std::string();
        const std::string lr_path = out_dir + "lena_lr.png";
        const std::string bicubic_path = out_dir + "lena_bicubic_4k.png";
        const std::string sr_path =
            (output.find('/') == std::string::npos) ? out_dir + output : output;
        SaveRgb(lr_path, lr);
        SaveRgb(bicubic_path, bicubic);
        SaveRgb(sr_path, sr);
        std::cout << "Wrote " << lr_path << ", " << bicubic_path << ", " << sr_path
                  << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}

cc_library(
    name = "eigen",
    hdrs = glob([
        "Eigen/*",
        "Eigen/**/*.h",
        "unsupported/Eigen/*",
        "unsupported/Eigen/**/*.h",
    ]),
    copts = ["-std=c++17"] + select({
        "@platforms//os:osx": [
            "-isystem",
            "/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers",
        ],
        "//conditions:default": [],
    }),
    defines = ["EIGEN_USE_BLAS=1"],
    includes = ["."],
    linkopts = select({
        "@platforms//os:osx": [
            "-framework",
            "Accelerate",
        ],
        "//conditions:default": [
            "-L/usr/lib/x86_64-linux-gnu",
            "-lopenblas",
            "-llapack",
            "-lm",
            "-lpthread",
        ],
    }),
    visibility = ["//visibility:public"],
)

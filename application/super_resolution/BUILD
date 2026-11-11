load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

cc_library(
    name = "tv_sr_admm",
    srcs = ["tv_sr_admm.cc"],
    hdrs = ["tv_sr_admm.h"],
    deps = ["@eigen//:eigen"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "super_resolution",
    srcs = ["super_resolution.cc"],
    data = ["testdata/lena.png"],
    deps = [
        ":tv_sr_admm",
        "//third_party/stb",
        "@eigen//:eigen",
    ],
)

cc_test(
    name = "tv_sr_admm_test",
    size = "small",
    srcs = ["tv_sr_admm_test.cc"],
    deps = [
        ":tv_sr_admm",
        "@eigen//:eigen",
        "@googletest//:gtest_main",
    ],
)

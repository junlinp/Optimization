load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
    name = "rotation_averaging",
    srcs = [
        "rotation_averaging.cc",
        "shonan_certificate.cc",
        "shonan_certificate.h",
    ],
    hdrs = ["rotation_averaging.h"],
    visibility = ["//visibility:public"],
    deps = ["@eigen//:eigen"],
)

cc_test(
    name = "rotation_averaging_test",
    srcs = ["rotation_averaging_test.cc"],
    deps = [
        ":rotation_averaging",
        "@eigen//:eigen",
        "@googletest//:gtest_main",
    ],
)

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

cc_library(
    name = "simulator",
    srcs = ["simulator.cc"],
    hdrs = ["simulator.h"],
    deps = [
        "@eigen//:eigen",
        "@sophus//:sophus",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "simulator_test",
    srcs = ["simulator_test.cc"],
    deps = [
        ":simulator",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "eskf_attitude",
    srcs = ["eskf_attitude.cc"],
    hdrs = ["eskf_attitude.h"],
    deps = [
        "@eigen//:eigen",
        "@sophus//:sophus",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "eskf_attitude_test",
    srcs = ["eskf_attitude_test.cc"],
    deps = [
        ":eskf_attitude",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "benchmark",
    srcs = ["benchmark.cc"],
    deps = [
        ":eskf_attitude",
        ":simulator",
        "@eigen//:eigen",
    ],
)

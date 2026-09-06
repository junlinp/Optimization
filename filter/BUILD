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

cc_library(
    name = "eqf_attitude",
    srcs = ["eqf_attitude.cc"],
    hdrs = ["eqf_attitude.h"],
    deps = [
        "@eigen//:eigen",
        "@sophus//:sophus",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "eqf_attitude_test",
    srcs = ["eqf_attitude_test.cc"],
    deps = [
        ":eqf_attitude",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "eqf_star_attitude",
    srcs = ["eqf_star_attitude.cc"],
    hdrs = ["eqf_star_attitude.h"],
    deps = [
        ":eqf_attitude",
        "@eigen//:eigen",
        "@sophus//:sophus",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "eqf_star_attitude_test",
    srcs = ["eqf_star_attitude_test.cc"],
    deps = [
        ":eqf_star_attitude",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "benchmark",
    srcs = ["benchmark.cc"],
    deps = [
        ":eqf_attitude",
        ":eqf_star_attitude",
        ":eskf_attitude",
        ":simulator",
        "@eigen//:eigen",
    ],
)

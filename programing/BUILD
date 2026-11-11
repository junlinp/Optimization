load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

cc_library(
    name = "linear_programing",
    srcs = [
        "linear_programing.cc",
        "linear_programing_optimized.cc",
        "mps_reader.cc",
    ],
    hdrs = [
        "linear_programing.h",
        "mps_problem.h",
    ],
    deps = [
        ":jordan_conic",
        "@eigen//:eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "solve_mps",
    srcs = ["solve_mps.cc"],
    deps = [
        ":jordan_conic",
        ":linear_programing",
        "@eigen//:eigen",
    ],
)

cc_test(
    name = "linear_solver_test",
    srcs = ["linear_solver_test.cpp"],
    deps = [
        ":linear_programing",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "mps_benchmark_test",
    srcs = ["mps_benchmark_test.cc"],
    data = glob(["testdata/**/*.mps"]),
    size = "medium",
    deps = [
        ":linear_programing",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "quadratic_programing",
    srcs = ["quadratic_programing.cc"],
    hdrs = ["quadratic_programing.h"],
    deps = [
        "@eigen//:eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "quadratic_solver_test",
    srcs = ["quadratic_solver_test.cpp"],
    deps = [
        ":quadratic_programing",
        "@googletest//:gtest_main",
    ],
)


cc_library(
    name = "jordan_conic",
    srcs = ["jordan_conic.cc", "jordan_core.inc"],
    hdrs = ["jordan_conic.h", "conic_optimality.h"],
    deps = ["@eigen//:eigen"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "jordan_conic_test",
    srcs = ["jordan_conic_test.cc"],
    deps = [":jordan_conic", "@googletest//:gtest_main"],
)

cc_test(
    name = "jordan_conic_dataset_test",
    srcs = ["jordan_conic_dataset_test.cc"],
    deps = [":jordan_conic", "@googletest//:gtest_main"],
)

cc_binary(
    name = "jordan_benchmark_cli",
    srcs = ["jordan_benchmark_cli.cc"],
    deps = [":jordan_conic"],
)

cc_test(
    name = "external_conic_benchmark_test",
    srcs = ["external_conic_benchmark_test.cc"],
    deps = ["@googletest//:gtest_main"],
    data = ["external_conic_benchmark.py", ":jordan_benchmark_cli", ":solve_mps"] + glob([
        "testdata/external/netlib/*.mps",
        "testdata/external/sdplib/*.dat-s",
        "testdata/external/cblib/sambal.cbf",
        "testdata/external/cblib/gbd.cbf",
        "testdata/external/cblib/nvs03.cbf",
    ]),
    size = "medium",
)

cc_test(
    name = "jordan_lp_backend_test",
    srcs = ["jordan_lp_backend_test.cc"],
    data = glob(["testdata/**/*.mps"]),
    deps = [":jordan_conic", ":linear_programing", "@googletest//:gtest_main"],
    size = "large",
)

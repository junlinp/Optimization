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
        "@eigen//:eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "solve_mps",
    srcs = ["solve_mps.cc"],
    deps = [
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
 
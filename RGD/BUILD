load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
    name = "rgd",
    srcs = ["rgd.cc"],
    hdrs = [
        "gradient_checker.h",
        "manifold.h",
        "rgd.h",
        "rgd_cost_function_interface.h",
        "so3_cost_function_interface.h",
    ],
    deps = ["@eigen//:eigen"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "sophus_manifold",
    hdrs = ["sophus_manifold.h"],
    deps = [
        "@eigen//:eigen",
        "@sophus//:sophus",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "sophus_manifold_test",
    srcs = ["sophus_manifold_test.cc"],
    deps = [
        ":rgd",
        ":sophus_manifold",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "rgd_test",
    srcs = ["rgd_test.cc"],
    deps = [
        ":rgd",
        "@googletest//:gtest_main",
        "@ceres//:ceres",
    ],
)


cc_library(
    name = "trust_region_subproblem",
    srcs = ["trust_region_subproblem.cc"],
    hdrs = [
        "trust_region_subproblem.h",
    ],
    deps = [
    	 "@eigen//:eigen",
	 ":conjugate_gradient",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "trust_region_subproblem_test",
    srcs = ["trust_region_subproblem_test.cc"],
    deps = [
        ":trust_region_subproblem",
        "@googletest//:gtest_main",
    ]
)


cc_library(
    name = "conjugate_gradient",
    srcs = ["conjugate_gradient.cc"],
    hdrs = [
        "conjugate_gradient.h",
    ],
    deps = [
        "@eigen//:eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "conjugate_gradient_test",
    srcs = ["conjugate_gradient_test.cc"],
    deps = [
        ":conjugate_gradient",
        "@googletest//:gtest_main",
    ]
)

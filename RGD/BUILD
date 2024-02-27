
cc_library(
    name = "rgd",
    srcs = ["rgd.cc"],
    hdrs = [
        "manifold.h",
        "rgd.h",
        "so3_cost_function_interface.h",

    ],
    deps = ["@com_gitlab_libeigen_eigen//:eigen"],
    visibility = ["//visibility:public"],
    )

cc_binary(
    name = "rgd_test",
    srcs = ["rgd_test.cc"],
    deps = [
        ":rgd",
        "@googletest//:gtest",
    ]
)

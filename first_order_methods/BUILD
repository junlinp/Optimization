load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
    name = "first_order_methods",
    srcs = ["gradient_descent.cc"],
    hdrs = [
        "first_oracle.h",
        "gradient_descent.h",
    ],
    deps = [
        "@eigen//:eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "first_order_methods_test",
    size = "medium",
    srcs = ["first_order_methods_test.cc"],
    deps = [
        ":first_order_methods",
        "@eigen//:eigen",
        "@googletest//:gtest",
    ],
)

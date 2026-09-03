load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
    name = "graph",
    srcs = [
        "graph.cc",
        "graph_normal_cut.cc",
    ],
    hdrs = [
        "graph.h",
        "graph_normal_cut.h",
    ],
    deps = [
        "@eigen//:eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "graph_test",
    srcs = ["graph_test_main.cc"],
    deps = [
        ":graph",
        "@googletest//:gtest",
    ],
)

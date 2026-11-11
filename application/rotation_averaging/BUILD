load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_test(
    name = "rotation_averaging_test",
    srcs = ["rotation_averaging_test.cc"],
    deps = [
        "@googletest//:gtest_main",
        "@eigen//:eigen",
	":rotation_averaging",
    ],
)

cc_library(
	name = "rotation_averaging",
	srcs = ["rotation_averaging.cc"],
	hdrs = ["rotation_averaging.h"],
	deps = [
	     "@eigen//:eigen",
	     "@ceres",
	],
	visibility = ["//visibility:public"],
)

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

cc_library(
    name = "euroc_types",
    hdrs = ["euroc_types.h"],
    deps = [
        "@eigen//:eigen",
        "@sophus//:sophus",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "yaml_lite",
    srcs = ["yaml_lite.cc"],
    hdrs = ["yaml_lite.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "euroc_loader",
    srcs = ["euroc_loader.cc"],
    hdrs = ["euroc_loader.h"],
    deps = [
        ":euroc_types",
        ":yaml_lite",
        "@eigen//:eigen",
        "@sophus//:sophus",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "image_io",
    srcs = ["image_io.cc"],
    hdrs = ["image_io.h"],
    deps = ["//third_party/stb"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "euroc_loader_test",
    srcs = ["euroc_loader_test.cc"],
    data = glob(["testdata/**"]),
    deps = [
        ":euroc_loader",
        ":euroc_types",
        ":yaml_lite",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "eskf_estimator",
    srcs = ["eskf_estimator.cc"],
    hdrs = ["eskf_estimator.h"],
    deps = [
        "@eigen//:eigen",
        "@sophus//:sophus",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "eskf_estimator_test",
    srcs = ["eskf_estimator_test.cc"],
    deps = [
        ":eskf_estimator",
        "@googletest//:gtest_main",
        "@sophus//:sophus",
    ],
)

cc_library(
    name = "triangulation",
    srcs = ["triangulation.cc"],
    hdrs = ["triangulation.h"],
    deps = ["@eigen//:eigen"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "stereo_rectifier",
    srcs = ["stereo_rectifier.cc"],
    hdrs = ["stereo_rectifier.h"],
    deps = [
        ":euroc_types",
        ":image_io",
        "@eigen//:eigen",
        "@sophus//:sophus",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "harris_corners",
    srcs = ["harris_corners.cc"],
    hdrs = ["harris_corners.h"],
    deps = [
        ":image_io",
        "@eigen//:eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "patch_matcher",
    srcs = ["patch_matcher.cc"],
    hdrs = ["patch_matcher.h"],
    deps = [":image_io"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "stereo_geometry_test",
    srcs = ["stereo_geometry_test.cc"],
    deps = [
        ":euroc_types",
        ":image_io",
        ":stereo_rectifier",
        ":triangulation",
        "@googletest//:gtest_main",
        "@sophus//:sophus",
    ],
)

cc_test(
    name = "harris_corners_test",
    srcs = ["harris_corners_test.cc"],
    deps = [
        ":harris_corners",
        ":image_io",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "patch_matcher_test",
    srcs = ["patch_matcher_test.cc"],
    deps = [
        ":image_io",
        ":patch_matcher",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "rigid_registration",
    srcs = ["rigid_registration.cc"],
    hdrs = ["rigid_registration.h"],
    deps = ["@eigen//:eigen"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "rigid_registration_test",
    srcs = ["rigid_registration_test.cc"],
    deps = [
        ":rigid_registration",
        "@googletest//:gtest_main",
        "@sophus//:sophus",
    ],
)

cc_library(
    name = "stereo_vo_frontend",
    srcs = ["stereo_vo_frontend.cc"],
    hdrs = ["stereo_vo_frontend.h"],
    deps = [
        ":harris_corners",
        ":image_io",
        ":patch_matcher",
        ":rigid_registration",
        ":stereo_rectifier",
        ":triangulation",
        "@eigen//:eigen",
        "@sophus//:sophus",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "vio_demo",
    srcs = ["vio_demo.cc"],
    deps = [
        ":eskf_estimator",
        ":euroc_loader",
        ":euroc_types",
        ":image_io",
        ":stereo_rectifier",
        ":stereo_vo_frontend",
        "@eigen//:eigen",
        "@sophus//:sophus",
    ],
)

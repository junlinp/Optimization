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
    name = "imu_trajectory_simulator",
    srcs = ["imu_trajectory_simulator.cc"],
    hdrs = ["imu_trajectory_simulator.h"],
    deps = [
        ":eskf_estimator",
        "@eigen//:eigen",
        "@sophus//:sophus",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "imu_trajectory_simulator_test",
    srcs = ["imu_trajectory_simulator_test.cc"],
    deps = [
        ":imu_trajectory_simulator",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "anees_benchmark",
    srcs = ["anees_benchmark.cc"],
    hdrs = ["anees_benchmark.h"],
    deps = [
        ":eskf_estimator",
        ":imu_trajectory_simulator",
        "@eigen//:eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "anees_benchmark_test",
    srcs = ["anees_benchmark_test.cc"],
    deps = [
        ":anees_benchmark",
        "@googletest//:gtest_main",
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

cc_library(
    name = "trajectory_evaluation",
    srcs = ["trajectory_evaluation.cc"],
    hdrs = ["trajectory_evaluation.h"],
    deps = [
        ":eskf_estimator",
        ":euroc_loader",
        ":euroc_types",
        ":harris_corners",
        ":image_io",
        ":patch_matcher",
        ":rigid_registration",
        ":stereo_rectifier",
        ":stereo_vo_frontend",
        "@eigen//:eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "trajectory_evaluation_test",
    srcs = ["trajectory_evaluation_test.cc"],
    data = glob(["testdata/**"]),
    deps = [
        ":trajectory_evaluation",
        "@googletest//:gtest_main",
        "@sophus//:sophus",
    ],
)

cc_binary(
    name = "vio_demo",
    srcs = ["vio_demo.cc"],
    deps = [
        ":euroc_loader",
        ":trajectory_evaluation",
        "@eigen//:eigen",
    ],
)

cc_binary(
    name = "euroc_benchmark",
    srcs = ["euroc_benchmark.cc"],
    deps = [
        ":eqvio_pipeline",
        ":euroc_loader",
        ":imu_trajectory_simulator",
        ":trajectory_evaluation",
    ],
)

cc_binary(
    name = "imu_mc_benchmark",
    srcs = ["imu_mc_benchmark.cc"],
    deps = [
        ":anees_benchmark",
        ":imu_trajectory_simulator",
    ],
)

cc_library(
    name = "eqvio",
    srcs = [
        "eqvio.cc",
        "eqvio_lie_groups.cc",
    ],
    hdrs = [
        "eqvio.h",
        "eqvio_lie_groups.h",
    ],
    deps = [
        ":eskf_estimator",
        "@eigen//:eigen",
        "@sophus//:sophus",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "vio_simulator",
    srcs = ["vio_simulator.cc"],
    hdrs = ["vio_simulator.h"],
    deps = [
        ":eqvio",
        ":imu_trajectory_simulator",
        "@eigen//:eigen",
        "@sophus//:sophus",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "eqvio_test",
    srcs = ["eqvio_test.cc"],
    deps = [
        ":eqvio",
        ":eskf_estimator",
        ":vio_simulator",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "eqvio_sim_benchmark",
    srcs = ["eqvio_sim_benchmark.cc"],
    deps = [
        ":eqvio",
        ":eskf_estimator",
        ":vio_simulator",
    ],
)

cc_library(
    name = "mono_feature_tracker",
    srcs = ["mono_feature_tracker.cc"],
    hdrs = ["mono_feature_tracker.h"],
    deps = [
        ":euroc_types",
        ":harris_corners",
        ":image_io",
        ":patch_matcher",
        ":stereo_rectifier",
        "@eigen//:eigen",
        "@sophus//:sophus",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "eqvio_pipeline",
    srcs = ["eqvio_pipeline.cc"],
    hdrs = ["eqvio_pipeline.h"],
    deps = [
        ":eqvio",
        ":eskf_estimator",
        ":euroc_types",
        ":image_io",
        ":mono_feature_tracker",
        ":trajectory_evaluation",
        "@eigen//:eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "mono_feature_tracker_test",
    srcs = ["mono_feature_tracker_test.cc"],
    deps = [
        ":mono_feature_tracker",
        ":stereo_rectifier",
        "@googletest//:gtest_main",
    ],
)

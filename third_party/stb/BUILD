load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "stb",
    srcs = ["stb_impl.cc"],
    hdrs = [
        "stb_image.h",
        "stb_image_write.h",
    ],
    includes = ["."],
    copts = ["-Wno-deprecated-declarations"],
    visibility = ["//visibility:public"],
)

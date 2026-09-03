load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
name = "cgraph",
srcs = glob(["src/**/*.cpp"]),
hdrs = glob(["src/**/*.h", "src/**/*.inl"]),
copts= ["-std=c++17"],
visibility = ["//visibility:public"],
)
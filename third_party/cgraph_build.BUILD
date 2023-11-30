cc_library(
name = "cgraph",
srcs = glob(["src/**/*.cpp"]),
hdrs = glob([
        "src/*",
        "src/**/*.h",
        "src/**/*.inl",
     ]),
copts= ["-std=c++17"],
defines=["_ENABLE_LIKELY_"],
visibility = ["//visibility:public"],
)
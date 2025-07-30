cc_library(
    name = "openblas",
    srcs = [],
    hdrs = [],
    includes = ["include"],
    linkopts = [
        "-L/usr/lib/x86_64-linux-gnu",
        "-lopenblas",
        "-llapack",
        "-lgfortran",
        "-lm",
        "-lpthread",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "blas",
    deps = [":openblas"],
    visibility = ["//visibility:public"],
) 
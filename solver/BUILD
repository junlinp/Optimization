cc_library(
	name = "lm",
	srcs = ["lm.cc"],
	hdrs = [
		"lm.h",
		  ],
    deps = [
        "@eigen//:eigen",
        "@openblas//:openblas",
    ]
)

cc_test(
    name = "solver_test",
    srcs = ["solver_test.cc"],
    deps = [
        ":lm",
        "@googletest//:gtest_main",
    ]
)

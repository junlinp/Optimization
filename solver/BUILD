cc_library(
	name = "lm",
	srcs = ["lm.cc"],
	hdrs = [
		"lm.h",
		  ],
    deps = [
        "@com_gitlab_libeigen_eigen//:eigen", 
        ]
)

cc_binary(
    name = "solver_test",
    srcs = ["solver_test.cc"],
    deps = [
        "lm",
        "@googletest//:gtest",
    ]
)

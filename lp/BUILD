
cc_library(
name = "linear_programing",
srcs = [
     "linear_programing.cc",
     ],
hdrs = [
"linear_programing.h",
],
deps = [
"@com_gitlab_libeigen_eigen//:eigen",
]
)

cc_test(
name = "linear_programing_test",
srcs = ["linear_programing_test.cc"],
deps = [
":linear_programing",
"@googletest//:gtest_main",
],
)

cc_library(
	name = "eigen",
	hdrs = glob([
		"Eigen/*",
		"Eigen/**/*.h",
		"unsupported/Eigen/*",
		"unsupported/Eigen/**/*.h",
	]),
	defines = ["EIGEN_USE_BLAS=1"],
	includes = ["."],
	copts = ["-std=c++17"],
	deps = ["@openblas//:openblas"],
	visibility = ["//visibility:public"]
)




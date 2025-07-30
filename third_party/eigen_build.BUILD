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
	linkopts = [
		"-L/usr/lib/x86_64-linux-gnu",
		"-lopenblas",
		"-llapack",
		"-lm",
		"-lpthread",
	],
	visibility = ["//visibility:public"]
)




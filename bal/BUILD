
cc_library(
	name = "load_problem",
	srcs = ["load_problem.cc"],
	hdrs = [
		"load_problem.h",
		 "problem.h",
		  "cost_function_auto.h"
		  ],
    deps = ["@com_gitlab_libeigen_eigen//:eigen"]
)

cc_library(
	name = "ceres_bal_solver",
	srcs = ["ceres_bal_solver.cc"],
	hdrs = [
		"bal_solver.h",
	 "ceres_bal_solver.h",
	 "problem.h",
	 "cost_function_auto.h",
	 ],
	deps = ["@ceres", "@com_gitlab_libeigen_eigen//:eigen"]
)

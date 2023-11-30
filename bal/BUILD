
cc_library(
	name = "load_problem",
	srcs = ["load_problem.cc"],
	hdrs = [
		"load_problem.h",
		 "problem.h",
		  "cost_function_auto.h"
		  ],
    deps = ["@com_gitlab_libeigen_eigen//:eigen", "@ceres"]
)

cc_library(
	name = "problem",
	srcs = [],
	hdrs = [
		"problem.h",
		"cost_function_auto.h",
	],
	deps = [
		"@com_gitlab_libeigen_eigen//:eigen",
		"@ceres",
	]
)
cc_library(
	name = "bal_solver",
	srcs = [],
	hdrs = ["bal_solver.h"],
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

cc_library(
	name = "daba_subproblem",
	srcs = ["daba_subproblem.cc"],
	hdrs = ["daba_subproblem.h"],
	deps = [
		"problem",
		"@ceres",
	]
)
cc_library(
	name = "daba_bal_solver",
	srcs = ["daba_bal_solver.cc"],
	hdrs = [
		"daba_bal_solver.h"
	],
	deps = [
		"bal_solver",
		"problem",
		"daba_subproblem",
		"//Graph:graph",
		]
)

cc_library(
	name = "daba_subproblem_manager",
	srcs = ["daba_subproblem_manager.cc"],
	hdrs = [
		"daba_subproblem_manager.h"
	],
	deps = [
		"daba_bal_solver",
		"@com_libcgraph//:cgraph",
	]
)
cc_binary(
	name ="cgraph_test",
	srcs = ["cgraph_test.cc"],
	deps = [
		"@com_libcgraph//:cgraph",
	]
)

cc_library(
	name = "evaluate",
	srcs = ["evaluate.cc"],
	hdrs = [
		"evaluate.h",
	],
	deps = [
		"@com_gitlab_libeigen_eigen//:eigen",
		"problem",
	]
)

cc_binary(
	name = "bal_main",
	srcs = ["bal_main.cc"],
	deps = [
		"daba_subproblem_manager",
		"load_problem",
		"ceres_bal_solver",
	]
)

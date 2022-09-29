load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")


def deps():

    maybe(
		new_git_repository,
		name="com_gitlab_libeigen_eigen",
		commit = "13b69fc1b0d1f7f0cbce70230c9a8794fcaa53b7",
		remote = "https://gitlab.com/libeigen/eigen.git",
		build_file="//third_party:eigen_build.BUILD"
	)


    maybe(
		git_repository,
		name="com_github_gflags_gflags",
		commit = "a738fdf9338412f83ab3f26f31ac11ed3f3ec4bd",
		remote = "https://github.com/gflags/gflags.git",
	)

    maybe(
		git_repository,
		name="com_github_google_glog",
		commit = "05fbc65278db1aa545ca5cb743c31bc717a48d0f",
		remote = "https://github.com/google/glog.git",
	)

    maybe(
		git_repository,
		name = "ceres",
		commit = "a762baa27529e50fc41a37824033245a6f0e9392",
		remote ="https://github.com/junlinp/ceres-solver.git"
	)
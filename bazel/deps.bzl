load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")


def deps():
    maybe(
        new_git_repository,
        name="com_libcgraph",
        commit="b499e5e79326b0bde09b268defcaff8cef733a90",
        remote="https://github.com/junlinp/CGraph.git",
        build_file="//third_party:cgraph_build.BUILD"
    )

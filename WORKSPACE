workspace(name="Optimization")

load("//bazel:deps.bzl", "deps")

deps()

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# http_archive(
#     name = "rules_m4",
#     urls = ["https://github.com/jmillikin/rules_m4/releases/download/v0.2/rules_m4-v0.2.tar.xz"],
#     sha256 = "c67fa9891bb19e9e6c1050003ba648d35383b8cb3c9572f397ad24040fb7f0eb",
# )
# load("@rules_m4//m4:m4.bzl", "m4_register_toolchains")
# m4_register_toolchains()

# http_archive(
#     name = "rules_flex",
#     urls = ["https://github.com/jmillikin/rules_flex/releases/download/v0.2/rules_flex-v0.2.tar.xz"],
#     sha256 = "f1685512937c2e33a7ebc4d5c6cf38ed282c2ce3b7a9c7c0b542db7e5db59d52",
# )
# load("@rules_flex//flex:flex.bzl", "flex_register_toolchains")
# flex_register_toolchains()

# http_archive(
#     name = "rules_bison",
#     urls = ["https://github.com/jmillikin/rules_bison/releases/download/v0.2.1/rules_bison-v0.2.1.tar.xz"],
#     sha256 = "9577455967bfcf52f9167274063ebb74696cb0fd576e4226e14ed23c5d67a693",
# )

# load("@rules_bison//bison:bison.bzl", "bison_register_toolchains")
# bison_register_toolchains()

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_proto_grpc",
    sha256 = "9ba7299c5eb6ec45b6b9a0ceb9916d0ab96789ac8218269322f0124c0c0d24e2",
    strip_prefix = "rules_proto_grpc-4.5.0",
    urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/releases/download/4.5.0/rules_proto_grpc-4.5.0.tar.gz"],
)

load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_toolchains", "rules_proto_grpc_repos")
rules_proto_grpc_toolchains()
rules_proto_grpc_repos()

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
rules_proto_dependencies()
rules_proto_toolchains()

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "gflags",
    urls = ["https://github.com/gflags/gflags/archive/refs/tags/v2.2.2.tar.gz"],
    strip_prefix = "gflags-2.2.2",
    sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
)

http_archive(
    name = "com_github_glog",
    repo_mapping = {
        "@com_github_gflags_gflags" : "@gflags",
    },
    sha256 = "f28359aeba12f30d73d9e4711ef356dc842886968112162bc73002645139c39c",
    strip_prefix = "glog-0.4.0",
    url = "https://github.com/google/glog/archive/v0.4.0.tar.gz",
)

# Note for any future version gate: 5.0.1 continues the 3.5 development line,
# so the legacy triple still reads WORLD=3, MAJOR=5, MINOR=0, PATCH=1.
# EIGEN_VERSION_AT_LEAST(5, 0, 0) is therefore false; check
# EIGEN_VERSION_STRING, which is "5.0.1".
http_archive(
    name = "eigen",
    build_file = "@Optimization//third_party:eigen_build.BUILD",
    sha256 = "e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec",
    strip_prefix = "eigen-5.0.1",
    url = "https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz",
)

# Ceres 2.2.0 rather than 2.0.0: 2.0.0 uses Eigen::MappedSparseMatrix, which
# Eigen removed in 5.0.
http_archive(
    name = "ceres",
    repo_mapping = {
        "@com_github_gflags_gflags": "@gflags",
        "@com_github_google_glog": "@com_github_glog",
        "@com_gitlab_libeigen_eigen": "@eigen",
    },
    sha256 = "1fc28e22ce190ce4c1db04d0c2ddfdf8dc836609866d09372ea8533c8b6d490b",
    strip_prefix = "ceres-solver-2.2.0",
    # The 2.2.0 tarball ships a stale Bazel source list: it names
    # parallel_for.cc, which no longer exists, and omits
    # parallel_vector_ops.cc, which does. One substitution fixes both.
    patch_cmds = [
        "sed -i.bak 's|\"parallel_for.cc\",|\"parallel_vector_ops.cc\",|' bazel/ceres.bzl",
    ],
    url = "https://github.com/ceres-solver/ceres-solver/archive/2.2.0.zip",
)

http_archive(
    name = "sophus",
    build_file = "@Optimization//third_party:sophus_build.BUILD",
    sha256 = "eb1da440e6250c5efc7637a0611a5b8888875ce6ac22bf7ff6b6769bbc958082",
    strip_prefix = "Sophus-1.22.10",
    url = "https://github.com/strasdat/Sophus/archive/refs/tags/1.22.10.tar.gz",
)

http_archive(
    name = "googletest",
    urls = ["https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz"],
    strip_prefix = "googletest-1.15.2",
    sha256 = "7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926",
)

# OpenBLAS configuration - using system libraries directly

# Hedron's Compile Commands Extractor for Bazel
# https://github.com/hedronvision/bazel-compile-commands-extractor
http_archive(
    name = "hedron_compile_commands",

    # Replace the commit hash (daae6f40adfa5fdb7c89684cbe4d88b691c63b2d) in both places (below) with the latest (https://github.com/hedronvision/bazel-compile-commands-extractor/commits/main), rather than using the stale one here.
    # Even better, set up Renovate and let it do the work for you (see "Suggestion: Updates" in the README).
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/daae6f40adfa5fdb7c89684cbe4d88b691c63b2d.tar.gz",
    strip_prefix = "bazel-compile-commands-extractor-daae6f40adfa5fdb7c89684cbe4d88b691c63b2d",
    # When you first run this tool, it'll recommend a sha256 hash to put here with a message like: "DEBUG: Rule 'hedron_compile_commands' indicated that a canonical reproducible form can be obtained by modifying arguments sha256 = ..."
)
load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
hedron_compile_commands_setup()
# bazel run @hedron_compile_commands//:refresh_all


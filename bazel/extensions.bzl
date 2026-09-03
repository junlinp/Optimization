load("//bazel:deps.bzl", "deps")

def _non_bcr_deps_impl(module_ctx):
    deps()

non_bcr_deps = module_extension(implementation = _non_bcr_deps_impl)

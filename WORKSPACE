workspace(name="Optimization")

load("//bazel:deps.bzl", "deps")

deps()

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_m4",
    urls = ["https://github.com/jmillikin/rules_m4/releases/download/v0.2/rules_m4-v0.2.tar.xz"],
    sha256 = "c67fa9891bb19e9e6c1050003ba648d35383b8cb3c9572f397ad24040fb7f0eb",
)
load("@rules_m4//m4:m4.bzl", "m4_register_toolchains")
m4_register_toolchains()

http_archive(
    name = "rules_flex",
    urls = ["https://github.com/jmillikin/rules_flex/releases/download/v0.2/rules_flex-v0.2.tar.xz"],
    sha256 = "f1685512937c2e33a7ebc4d5c6cf38ed282c2ce3b7a9c7c0b542db7e5db59d52",
)
load("@rules_flex//flex:flex.bzl", "flex_register_toolchains")
flex_register_toolchains()

http_archive(
    name = "rules_bison",
    urls = ["https://github.com/jmillikin/rules_bison/releases/download/v0.2.1/rules_bison-v0.2.1.tar.xz"],
    sha256 = "9577455967bfcf52f9167274063ebb74696cb0fd576e4226e14ed23c5d67a693",
)

load("@rules_bison//bison:bison.bzl", "bison_register_toolchains")
bison_register_toolchains()


load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "hedron_compile_commands",
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/12440540f4461cac006a357257d698aed74a2423.tar.gz",
    strip_prefix = "bazel-compile-commands-extractor-12440540f4461cac006a357257d698aed74a2423",
)
load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
hedron_compile_commands_setup()
load("@rules_flex//flex:flex.bzl", "flex_cc_library")
load("@rules_bison//bison:bison.bzl", "bison_cc_library")
load("@rules_bison//bison:bison.bzl", "bison")

genrule(
    name = "parse_source",
    srcs = ["mps.yy"],
    outs = ["parser.cc"],
    cmd = "M4=$(M4) $(BISON) --output $@ $<",
    toolchains = [
        "@rules_bison//bison:current_bison_toolchain",
        "@rules_m4//m4:current_m4_toolchain",
    ],
)

genrule(
    name = "parse_headers",
    srcs = ["mps.yy"],
    outs = ["parser.h"],
    cmd = "M4=$(M4) $(BISON) --defines=$@ $<",
    toolchains = [
        "@rules_bison//bison:current_bison_toolchain",
        "@rules_m4//m4:current_m4_toolchain",
    ],
)

genrule(
    name = "lex_source",
    srcs = ["mps.ll"],
    outs = ["lex.cc"],
    cmd = "M4=$(M4) $(FLEX) --outfile=$@ $<",
    toolchains = [
        "@rules_flex//flex:current_flex_toolchain",
        "@rules_m4//m4:current_m4_toolchain",
    ],
)

cc_library(
    name = "mps_format_io",
    srcs = ["mps_format_io.cc"],
    hdrs = ["mps_format_io.h"],
)
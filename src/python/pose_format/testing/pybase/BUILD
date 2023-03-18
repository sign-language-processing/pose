licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

py_library(
    name = "pybase",
    srcs = ["pybase_dummy.py"],
    srcs_version = "PY2AND3",
)

genrule(
    name = "dummy",
    outs = ["pybase_dummy.py"],
    cmd = "touch $@",
    visibility = ["//visibility:private"],
)
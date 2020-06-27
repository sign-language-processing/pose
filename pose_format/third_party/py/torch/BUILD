licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

py_library(
    name = "pytorch",
    srcs = ["torch_dummy.py"],
    srcs_version = "PY2AND3",
)

genrule(
    name = "dummy",
    outs = ["torch_dummy.py"],
    cmd = "touch $@",
    visibility = ["//visibility:private"],
)
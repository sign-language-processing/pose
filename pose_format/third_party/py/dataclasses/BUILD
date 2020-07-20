licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

py_library(
    name = "dataclasses",
    srcs = ["dataclasses_dummy.py"],
    srcs_version = "PY2AND3",
)

genrule(
    name = "dummy",
    outs = ["dataclasses_dummy.py"],
    cmd = "touch $@",
    visibility = ["//visibility:private"],
)
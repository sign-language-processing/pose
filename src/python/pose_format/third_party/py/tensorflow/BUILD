licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

py_library(
    name = "tensorflow",
    srcs = ["tensorflow_dummy.py"],
    srcs_version = "PY2AND3",
)

genrule(
    name = "dummy",
    outs = ["tensorflow_dummy.py"],
    cmd = "touch $@",
    visibility = ["//visibility:private"],
)
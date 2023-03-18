licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

py_library(
    name = "tqdm",
    srcs = ["tqdm_dummy.py"],
    srcs_version = "PY2AND3",
)

genrule(
    name = "dummy",
    outs = ["tqdm_dummy.py"],
    cmd = "touch $@",
    visibility = ["//visibility:private"],
)
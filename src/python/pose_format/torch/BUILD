licenses(["notice"])

py_library(
    name = "pose_body",
    srcs = ["pose_body.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/numpy",
        "//:pose_body",
        "//:pose_header",
        "//torch/masked:tensor",
        "//utils:reader",
        "//third_party/py/torch:pytorch",
    ],
)

py_library(
    name = "pose_representation",
    srcs = ["pose_representation.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//:pose_header",
        "//:pose_representation",
        "//third_party/py/torch:pytorch",
    ],
)

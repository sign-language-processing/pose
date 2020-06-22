licenses(["notice"])

py_library(
    name = "pose_body",
    srcs = ["pose_body.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/pose_format:pose_body",
        "//third_party/py/pose_format:pose_header",
        "//third_party/py/pose_format/torch/masked:tensor",
        "//third_party/py/pose_format/utils:reader",
        "//third_party/py/torch:pytorch",
    ],
)

py_library(
    name = "pose_representation",
    srcs = ["pose_representation.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/pose_format:pose_header",
        "//third_party/py/pose_format:pose_representation",
        "//third_party/py/torch:pytorch",
    ],
)

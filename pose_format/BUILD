# This library includes utilities to handle pose data

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "pose",
    srcs = ["pose.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":pose_body",
        ":pose_header",
        "//third_party/py/numpy",
        "//third_party/py/pose_format/numpy:pose_body",
        "//third_party/py/pose_format/utils:fast_math",
        "//third_party/py/pose_format/utils:reader",
    ],
)

py_library(
    name = "pose_body",
    srcs = ["pose_body.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":pose_header",
        "//third_party/py/numpy",
        "//third_party/py/pose_format/utils:reader",
    ],
)

py_library(
    name = "pose_header",
    srcs = ["pose_header.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/pose_format/utils:reader",
    ],
)

py_library(
    name = "pose_visualizer",
    srcs = ["pose_visualizer.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":pose",
        "//third_party/OpenCVX:core",
        "//third_party/OpenCVX:video",
        "//third_party/py/numpy",
        "//third_party/py/tqdm",
    ],
)

py_library(
    name = "pose_representation",
    srcs = ["pose_representation.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":pose_header",
    ],
)

py_test(
    name = "test_pose",
    srcs = ["test_pose.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":pose",
    ],
)

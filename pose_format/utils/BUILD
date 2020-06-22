licenses(["notice"])

py_library(
    name = "fast_math",
    srcs = ["fast_math.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "openpose",
    srcs = ["openpose.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/pose_format:pose",
        "//third_party/py/pose_format:pose_body",
        "//third_party/py/pose_format:pose_header",
    ],
)

py_library(
    name = "reader",
    srcs = ["reader.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/dataclasses",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/py/torch:pytorch",
    ],
)

py_test(
    name = "test_reader",
    srcs = ["test_reader.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":reader",
        "//testing/pybase",
    ],
)

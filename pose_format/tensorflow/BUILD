licenses(["notice"])

py_library(
    name = "pose_representation",
    srcs = ["pose_representation.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/pose_format:pose_representation",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "pose_representation_test",
    srcs = ["pose_representation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":pose_representation",
        "//testing/pybase",
        "//third_party/py/pose_format:pose_header",
        "//third_party/py/pose_format:pose_representation",
        "//third_party/py/pose_format/tensorflow/representation:angle",
        "//third_party/py/pose_format/tensorflow/representation:distance",
        "//third_party/py/pose_format/tensorflow/representation:inner_angle",
        "//third_party/py/pose_format/tensorflow/representation:point_line_distance",
        "//third_party/py/pose_format/utils:openpose",
        "//third_party/py/tensorflow",
    ],
)

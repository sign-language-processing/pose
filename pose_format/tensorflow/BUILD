licenses(["notice"])

py_library(
    name = "pose_representation",
    srcs = ["pose_representation.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//:pose_representation",
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
        "//:pose_header",
        "//:pose_representation",
        "//tensorflow/representation:angle",
        "//tensorflow/representation:distance",
        "//tensorflow/representation:inner_angle",
        "//tensorflow/representation:point_line_distance",
        "//utils:openpose",
        "//third_party/py/tensorflow",
    ],
)

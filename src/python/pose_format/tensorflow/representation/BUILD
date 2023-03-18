licenses(["notice"])

py_library(
    name = "angle",
    srcs = ["angle.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "angle_test",
    srcs = ["angle_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":angle",
        "//testing/pybase",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "distance",
    srcs = ["distance.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "distance_test",
    srcs = ["distance_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":distance",
        "//testing/pybase",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "inner_angle",
    srcs = ["inner_angle.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "inner_angle_test",
    srcs = ["inner_angle_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":inner_angle",
        "//testing/pybase",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "point_line_distance",
    srcs = ["point_line_distance.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":distance",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "point_line_distance_test",
    srcs = ["point_line_distance_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":point_line_distance",
        "//testing/pybase",
        "//third_party/py/tensorflow",
    ],
)

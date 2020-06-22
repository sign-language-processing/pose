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
    name = "test_angle",
    srcs = ["test_angle.py"],
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
    name = "test_distance",
    srcs = ["test_distance.py"],
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
    name = "test_inner_angle",
    srcs = ["test_inner_angle.py"],
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
    name = "test_point_line_distance",
    srcs = ["test_point_line_distance.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":point_line_distance",
        "//testing/pybase",
        "//third_party/py/tensorflow",
    ],
)

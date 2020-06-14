py_library(
    name = "angle",
    srcs = ["angle.py"],
	srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow"
    ],
    visibility = ["//visibility:public"]
)

py_test(
    name = "test_angle",
    srcs = ["test_angle.py"],
	srcs_version = "PY3",
    deps = [
    	":angle",
        "//third_party/py/tensorflow"
    ],
    visibility = ["//visibility:public"]
)

py_library(
    name = "distance",
    srcs = ["distance.py"],
	srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow"
    ],
    visibility = ["//visibility:public"]
)

py_test(
    name = "test_distance",
    srcs = ["test_distance.py"],
	srcs_version = "PY3",
    deps = [
    	":distance",
        "//third_party/py/tensorflow"
    ],
    visibility = ["//visibility:public"]
)

py_library(
    name = "inner_angle",
    srcs = ["inner_angle.py"],
	srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow"
    ],
    visibility = ["//visibility:public"]
)

py_test(
    name = "test_inner_angle",
    srcs = ["test_inner_angle.py"],
	srcs_version = "PY3",
    deps = [
    	":inner_angle",
        "//third_party/py/tensorflow"
    ],
    visibility = ["//visibility:public"]
)

py_library(
    name = "point_line_distance",
    srcs = ["point_line_distance.py"],
	srcs_version = "PY3",
    deps = [
    	":distance",
        "//third_party/py/tensorflow"
    ],
    visibility = ["//visibility:public"]
)

py_test(
    name = "test_point_line_distance",
    srcs = ["test_point_line_distance.py"],
	srcs_version = "PY3",
    deps = [
    	":point_line_distance",
        "//third_party/py/tensorflow"
    ],
    visibility = ["//visibility:public"]
)

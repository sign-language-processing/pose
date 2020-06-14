
py_library(
    name = "fast_math",
    srcs = ["fast_math.py"],
	srcs_version = "PY3",
    deps = [
        "//third_party/py/numpy"
    ],
    visibility = ["//visibility:public"]
)

py_library(
    name = "openpose",
    srcs = ["openpose.py"],
	srcs_version = "PY3",
    deps = [
	    "//:pose",
	    "//:pose_body",
	    "//:pose_header",
        "//third_party/py/numpy"
    ],
    visibility = ["//visibility:public"]
)

py_library(
    name = "reader",
    srcs = ["reader.py"],
	srcs_version = "PY3",
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/torch",
        "//third_party/py/tensorflow"
    ],
    visibility = ["//visibility:public"]
)

py_test(
    name = "test_reader",
    srcs = ["test_reader.py"],
	srcs_version = "PY3",
    deps = [
    	":reader"
    ],
    visibility = ["//visibility:public"]
)
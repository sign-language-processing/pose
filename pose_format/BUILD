# This library includes utilities to handle pose data


exports_files(["LICENSE"])


py_library(
    name = "pose",
    srcs = ["pose.py"],
	srcs_version = "PY3",
    deps = [
    	":pose_body",
    	":pose_header",
    	"//utils:reader",
    	"//utils:fast_math",
    	"//numpy:pose_body",
        "//third_party/py/numpy"
    ],
    visibility = ["//visibility:public"]
)

py_library(
    name = "pose_body",
    srcs = ["pose_body.py"],
	srcs_version = "PY3",
    deps = [
	    ":pose_header",
	    "//utils:reader",
        "//third_party/py/numpy"
    ],
    visibility = ["//visibility:public"]
)

py_library(
    name = "pose_header",
    srcs = ["pose_header.py"],
	srcs_version = "PY3",
    deps = [
	    "//utils:reader"
    ],
    visibility = ["//visibility:public"]
)

py_library(
    name = "pose_visualizer",
    srcs = ["pose_visualizer.py"],
	srcs_version = "PY3",
    deps = [
	    ":pose",
	    "//third_party/py/tqdm",
	    "//third_party/py/numpy",
	    "//third_party/py/cv2"
    ],
    visibility = ["//visibility:public"]
)

py_library(
    name = "pose_representation",
    srcs = ["pose_representation.py"],
	srcs_version = "PY3",
    deps = [
	    ":pose_header"
    ],
    visibility = ["//visibility:public"]
)

py_library(
    name = "pose_body",
    srcs = ["pose_body.py"],
	srcs_version = "PY3",
    deps = [
    	"//:pose_header",
    	"//:pose_body",
    	"//utils:reader",
    	"//torch/masked:tensor",
        "//third_party/py/torch",
        "//third_party/py/numpy"
    ],
    visibility = ["//visibility:public"]
)


py_library(
    name = "pose_representation",
    srcs = ["pose_representation.py"],
	srcs_version = "PY3",
    deps = [
    	"//:pose_header",
    	"//:pose_representation",
        "//third_party/py/torch"
    ],
    visibility = ["//visibility:public"]
)
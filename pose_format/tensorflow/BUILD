py_library(
    name = "pose_representation",
    srcs = ["pose_representation.py"],
	srcs_version = "PY3",
    deps = [
    	"//:pose_representation",
        "//third_party/py/tensorflow"
    ],
    visibility = ["//visibility:public"]
)

py_test(
    name = "test_pose_representation",
    srcs = ["test_pose_representation.py"],
	srcs_version = "PY3",
    deps = [
    	":pose_representation",
    	"//:pose_header",
    	"//tensorflow/representation",
    	"//utils:openpose",
        "//third_party/py/tensorflow"
    ],
    visibility = ["//visibility:public"]
)
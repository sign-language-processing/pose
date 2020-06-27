licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

py_library(
    name = "core",
    srcs = ["cv2_dummy.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "video",
    srcs = ["cv2_dummy.py"],
    srcs_version = "PY2AND3",
)

genrule(
    name = "dummy",
    outs = ["cv2_dummy.py"],
    cmd = "touch $@",
    visibility = ["//visibility:private"],
)
licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

py_library(
    name = "cv2",
    srcs = ["cv2_dummy.py"],
    srcs_version = "PY2AND3",
)

genrule(
    name = "dummy",
    outs = ["cv2_dummy.py"],
    cmd = "touch $@",
    visibility = ["//visibility:private"],
)
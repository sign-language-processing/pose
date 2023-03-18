licenses(["notice"])

py_library(
    name = "tensor",
    srcs = ["tensor.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "tensorflow",
    srcs = ["tensorflow.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":tensor",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "tensorflow_test",
    srcs = ["tensorflow_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":tensor",
        ":tensorflow",
        "//testing/pybase",
    ],
)

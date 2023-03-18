licenses(["notice"])

py_library(
    name = "tensor",
    srcs = ["tensor.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/torch:pytorch",
    ],
)

py_library(
    name = "torch",
    srcs = ["torch.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":tensor",
        "//third_party/py/torch:pytorch",
    ],
)

py_test(
    name = "torch_test",
    srcs = ["torch_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":tensor",
        ":torch",
        "//testing/pybase",
    ],
)

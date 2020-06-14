py_library(
    name = "tensor",
    srcs = ["tensor.py"],
	srcs_version = "PY3",
    deps = [
        "//third_party/py/torch"
    ],
    visibility = ["//visibility:public"]
)


py_library(
    name = "torch",
    srcs = ["torch.py"],
	srcs_version = "PY3",
    deps = [
    	":tensor",
        "//third_party/py/torch"
    ],
    visibility = ["//visibility:public"]
)


py_test(
    name = "test_torch",
    srcs = ["test_torch.py"],
	srcs_version = "PY3",
    deps = [
    	":torch",
    	":tensor"
    ],
    visibility = ["//visibility:public"]
)
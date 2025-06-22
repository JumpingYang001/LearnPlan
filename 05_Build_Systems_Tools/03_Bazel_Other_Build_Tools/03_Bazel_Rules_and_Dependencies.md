# Bazel Rules and Dependencies

## Overview
This section covers Bazel's built-in rules for C/C++, managing external dependencies, and using repository rules.

## C/C++ Example: Using External Dependency
```python
# BUILD file
cc_binary(
    name = "main",
    srcs = ["main.cpp"],
    deps = ["@mylib//:mylib"],
)
```
```python
# WORKSPACE file (snippet)
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "mylib",
    url = "https://example.com/mylib.tar.gz",
    strip_prefix = "mylib-1.0.0",
)
```

This example demonstrates how to use an external C++ library with Bazel.

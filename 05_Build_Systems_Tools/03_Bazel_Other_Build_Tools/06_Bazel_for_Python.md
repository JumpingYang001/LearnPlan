# Bazel for Python

## Overview
Learn how Bazel manages Python rules, pip dependencies, and packaging.

## Example: Python Binary with Bazel
```python
# BUILD file
py_binary(
    name = "hello_py",
    srcs = ["hello.py"],
)
```
```python
# hello.py
print("Hello from Bazel Python!")
```

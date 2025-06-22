# Bazel Basics

## Overview
Learn Bazel's core concepts, including workspaces, packages, targets, BUILD files, and WORKSPACE configuration. This section explains how to set up Bazel for C/C++ projects.

## C/C++ Example: Minimal Bazel Project
```python
# WORKSPACE file (empty for simple projects)
```
```python
# BUILD file
cc_binary(
    name = "hello_world",
    srcs = ["hello_world.cpp"],
)
```
```cpp
// hello_world.cpp
#include <iostream>
int main() {
    std::cout << "Hello, Bazel!" << std::endl;
    return 0;
}
```

This example shows a minimal Bazel setup for building a C++ binary.

# Bazel for C/C++

## Overview
Learn how to use Bazel's C/C++ rules, toolchains, compilation flags, and linking. This section provides a sample C++ project with Bazel.

## C/C++ Example: Multiple Source Files
```python
# BUILD file
cc_library(
    name = "mathlib",
    srcs = ["mathlib.cpp"],
    hdrs = ["mathlib.h"],
)
cc_binary(
    name = "main",
    srcs = ["main.cpp"],
    deps = [":mathlib"],
)
```
```cpp
// mathlib.h
#pragma once
int add(int a, int b);
```
```cpp
// mathlib.cpp
#include "mathlib.h"
int add(int a, int b) { return a + b; }
```
```cpp
// main.cpp
#include <iostream>
#include "mathlib.h"
int main() {
    std::cout << "2 + 3 = " << add(2, 3) << std::endl;
    return 0;
}
```

This example shows a Bazel C++ project with a library and a binary target.

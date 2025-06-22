# Project: Multi-language Monorepo

## Description
Build a monorepo structure with multiple languages and implement Bazel configuration for all components, including shared dependencies and cross-language integration.

## Example Structure
```
monorepo/
  WORKSPACE
  cpp_lib/
    BUILD
    math.cpp
    math.h
  java_app/
    BUILD
    Main.java
  python_app/
    BUILD
    main.py
```

## C++ Example: cpp_lib/BUILD
```python
cc_library(
    name = "math",
    srcs = ["math.cpp"],
    hdrs = ["math.h"],
)
```
```cpp
// math.h
#pragma once
int add(int a, int b);
```
```cpp
// math.cpp
#include "math.h"
int add(int a, int b) { return a + b; }
```

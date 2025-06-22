# Project 1: Cross-Platform Library

## Goal
Create a C/C++ library with a CMake build system that supports multiple platforms and compilers, and implements proper package export.

### Example: CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.15)
project(MyCrossPlatformLib)
add_library(mycpplib src/mycpplib.c)
target_include_directories(mycpplib PUBLIC include)
set_target_properties(mycpplib PROPERTIES EXPORT_NAME MyCppLib)
install(TARGETS mycpplib EXPORT MyCppLibTargets DESTINATION lib)
install(EXPORT MyCppLibTargets DESTINATION lib/cmake/MyCppLib)
```

```c
// src/mycpplib.c
#include "mycpplib.h"
int add(int a, int b) { return a + b; }
```

```c
// include/mycpplib.h
int add(int a, int b);
```

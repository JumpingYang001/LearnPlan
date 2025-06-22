# Project 2: Multi-Component Application

## Goal
Build a complex application with multiple components, manage internal and external dependencies, and create installation and packaging.

### Example: CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.15)
project(MultiComponentApp)
add_subdirectory(libfoo)
add_subdirectory(app)
```

// libfoo/CMakeLists.txt
```cmake
add_library(foo STATIC foo.c)
target_include_directories(foo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
```

// app/CMakeLists.txt
```cmake
add_executable(myapp main.c)
target_link_libraries(myapp PRIVATE foo)
```

```c
// libfoo/foo.c
#include "foo.h"
int foo() { return 42; }
```

```c
// libfoo/foo.h
int foo();
```

```c
// app/main.c
#include "foo.h"
#include <stdio.h>
int main() {
    printf("foo() = %d\n", foo());
    return 0;
}
```

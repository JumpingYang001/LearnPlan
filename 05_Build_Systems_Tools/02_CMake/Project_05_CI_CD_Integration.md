# Project 5: CI/CD Integration

## Goal
Set up a CMake project for continuous integration, implement multi-platform testing, and create automated deployment with CPack.

### Example: CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.15)
project(CICDProject)
add_executable(ciapp main.c)
enable_testing()
add_test(NAME RunApp COMMAND ciapp)
include(CPack)
set(CPACK_GENERATOR "ZIP")
```

```c
// main.c
#include <stdio.h>
int main() { printf("CI/CD Ready!\n"); return 0; }
```

# CMake Fundamentals

## Build System Concepts
- Build system vs. build system generator
- Build configurations (Debug, Release)
- Cross-platform considerations
- Dependency management

### Example: Minimal CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.10)
project(HelloCMake)
add_executable(hello main.c)
```

```c
// main.c
#include <stdio.h>
int main() {
    printf("Hello, CMake!\n");
    return 0;
}
```

## CMake Architecture
- CMake language and generator concept
- Build process stages
- CMakeCache and CMakeFiles

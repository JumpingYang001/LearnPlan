# Project 4: Cross-Compilation Toolchain

## Goal
Create toolchain files for different platforms, test cross-compilation workflow, and implement platform-specific optimizations.

### Example: toolchain-arm.cmake
```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
```

### Example: CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.15)
project(CrossCompileDemo)
add_executable(demo main.c)
```

```c
// main.c
int main() { return 0; }
```

# Building C/C++ Projects with CMake

## Project Structure
- Single vs. multi-directory projects
- Subprojects and subdirectories

### Example: Multi-directory Project
```cmake
add_subdirectory(src)
```

## Targets and Libraries
- add_executable, add_library

### Example: Static Library
```cmake
add_library(mylib STATIC mylib.c)
target_include_directories(mylib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
```

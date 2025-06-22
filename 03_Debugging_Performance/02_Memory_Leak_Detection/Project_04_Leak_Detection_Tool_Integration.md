# Project 4: Leak Detection Tool Integration

## Description
Create build system integration for leak detection tools and automate leak detection in CI/CD pipeline.

### Example: CMake Integration for ASAN
```cmake
add_executable(myapp main.cpp)
target_compile_options(myapp PRIVATE -fsanitize=address)
target_link_options(myapp PRIVATE -fsanitize=address)
```

### Example: GitHub Actions CI
```yaml
- name: Run ASAN
  run: ./myapp
```

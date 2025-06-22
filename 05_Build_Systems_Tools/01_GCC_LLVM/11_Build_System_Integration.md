# Build System Integration

## Autotools Integration
- Compiler detection, feature testing, cross-compilation.

## CMake Integration
- Toolchain files, feature detection, generator expressions.

## Other Build Systems
- Bazel, Meson, Ninja, custom systems.

## Compiler Wrappers
- ccache, distcc, sccache, launcher creation.

**CMake Example:**
```cmake
cmake_minimum_required(VERSION 3.10)
project(Example)
add_executable(main main.c)
```

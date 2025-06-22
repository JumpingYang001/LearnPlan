# Cross-Compilation and Toolchains

## Toolchain Files
- Structure and components
- Compiler selection

### Example: Toolchain File
```cmake
# toolchain-arm.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
```

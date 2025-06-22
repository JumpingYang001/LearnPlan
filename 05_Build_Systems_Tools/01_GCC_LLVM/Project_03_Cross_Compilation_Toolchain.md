# Project: Cross-Compilation Toolchain

## Goal
Build a complete cross-compilation toolchain supporting multiple target architectures.

## Example: Cross-Compiling (C)
```c
// Compile for ARM: arm-linux-gnueabihf-gcc -o hello hello.c
#include <stdio.h>
int main() {
    printf("Hello, ARM cross-compilation!\n");
    return 0;
}
```

## Toolchain Steps
- Build binutils
- Build GCC/Clang for target
- Set up sysroot and libraries

# Cross-Compilation and Toolchains

## Cross-Compilation Concepts
- Target triples, sysroot, library paths, ABI compatibility.

## Toolchain Construction
- Components, bootstrapping, Canadian cross.

## Toolchain Management
- Crosstool-NG, Buildroot, Yocto, Docker-based toolchains.

## Multilib and Multiarch
- Supporting multiple architectures, library organization, configuration.

**C Example:**
```c
// Example: cross-compiling for ARM
// arm-linux-gnueabihf-gcc -o hello hello.c
#include <stdio.h>
int main() {
    printf("Hello, ARM!\n");
    return 0;
}
```

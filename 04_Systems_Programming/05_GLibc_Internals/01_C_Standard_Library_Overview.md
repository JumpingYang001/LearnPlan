# C Standard Library Overview

## Description
Overview of the C standard library, its role, C11/C17 features, glibc's relationship with the kernel, and comparison with other implementations.

## Example: Simple printf and system call
```c
#include <stdio.h>
#include <unistd.h>

int main() {
    printf("Hello from glibc!\n");
    write(1, "Direct syscall!\n", 16);
    return 0;
}
```

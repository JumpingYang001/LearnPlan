# GCC-Specific Optimizations

## Overview
Explains GCC optimization flags, pragmas, attributes, and LTO.

## Example: GCC Attributes and Flags (C/C++)
```c
__attribute__((always_inline)) inline int fast_add(int a, int b) {
    return a + b;
}
```
// Compile with -O2 -flto for link-time optimization.

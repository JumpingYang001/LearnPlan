# MSVC Compiler Optimizations

## Overview
Discusses Visual C++ optimization options, directives, and whole program optimization (WPO).

## Example: MSVC Optimization (C/C++)
```c
__forceinline int fast_mul(int a, int b) {
    return a * b;
}
```
// Use /O2 and /GL for whole program optimization in MSVC.

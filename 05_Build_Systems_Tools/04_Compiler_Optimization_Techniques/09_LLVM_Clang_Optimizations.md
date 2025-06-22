# LLVM/Clang Optimizations

## Overview
Covers LLVM optimization passes, Clang features, and profile-guided optimization (PGO).

## Example: Profile-Guided Optimization (C/C++)
```c
int compute(int x) {
    if (x > 0) return x * 2;
    else return -x;
}
```
// Use clang with -fprofile-generate and -fprofile-use for PGO.

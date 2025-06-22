# Data-Level Parallelism

## Overview
Focuses on SIMD, auto-vectorization, and memory alignment.

## Example: SIMD Auto-Vectorization (C/C++)
```c
#include <immintrin.h>
void add_arrays(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
```
// Compile with -O2/-O3 and -ftree-vectorize. Use -march=native for best results.

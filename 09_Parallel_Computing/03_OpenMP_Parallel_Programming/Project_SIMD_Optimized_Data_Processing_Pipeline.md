# Project: SIMD-Optimized Data Processing Pipeline

## Description
Build a data processing pipeline using OpenMP SIMD. Implement vectorized mathematical operations. Create performance comparison with non-vectorized code.

## Example Code
```c
// Example: SIMD vectorized addition
#include <omp.h>
#include <stdio.h>
#define N 1024
float a[N], b[N], c[N];

void simd_add() {
    #pragma omp simd
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}
```

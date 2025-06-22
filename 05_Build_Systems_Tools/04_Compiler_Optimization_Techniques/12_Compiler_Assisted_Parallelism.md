# Compiler-Assisted Parallelism

## Overview
Covers OpenMP, auto-parallelization, and offloading to accelerators.

## Example: OpenMP Parallel For (C/C++)
```c
#include <omp.h>
void parallel_sum(int *a, int n) {
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        sum += a[i];
    }
}
```
// Compile with -fopenmp to enable OpenMP parallelism.

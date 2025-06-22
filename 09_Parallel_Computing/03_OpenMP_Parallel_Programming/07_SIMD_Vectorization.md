# SIMD Vectorization

## Description
Understand SIMD concepts, vectorization directives, alignment, padding, and implement vectorized code with OpenMP.

## Example
```c
// Example: SIMD vectorization in OpenMP
#include <omp.h>
#include <stdio.h>

int main() {
    float a[8], b[8], c[8];
    for (int i = 0; i < 8; i++) { a[i] = i; b[i] = 2*i; }
    #pragma omp simd
    for (int i = 0; i < 8; i++) {
        c[i] = a[i] + b[i];
    }
    for (int i = 0; i < 8; i++) printf("c[%d]=%f\n", i, c[i]);
    return 0;
}
```

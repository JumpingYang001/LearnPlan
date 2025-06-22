# Advanced Compiler Topics

## Whole Program Optimization
- LTO, IPO, implementation in GCC/LLVM.

## Polyhedral Optimization
- Loop transformation, dependency analysis, auto-parallelization.

## Vectorization
- Auto-vectorization, intrinsics, cost models, target-specifics.

## OpenMP and Auto-parallelization
- Compiler support, automatic parallelization, runtime libraries.

**C Example:**
```c
// Compile with: gcc -fopenmp -O2 example.c
#include <omp.h>
#include <stdio.h>
int main() {
    #pragma omp parallel for
    for (int i = 0; i < 4; ++i) {
        printf("Thread %d\n", omp_get_thread_num());
    }
    return 0;
}
```

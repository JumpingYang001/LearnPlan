# Project: Parallel Numerical Solver

## Description
Develop a parallel solver for differential equations. Implement domain decomposition techniques. Create performance analysis and visualization.

## Example Code
```c
// Example: Parallel Jacobi solver
#include <omp.h>
#include <stdio.h>
#define N 100
float u[N][N], unew[N][N];

void jacobi_step() {
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            unew[i][j] = 0.25f * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]);
        }
    }
}
```

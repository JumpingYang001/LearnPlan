# Project: Distributed Matrix Multiplication

## Description
Implement parallel matrix multiplication using MPI. Optimize communication patterns and benchmark against a sequential implementation.

## Example Code
```c
#include <mpi.h>
#include <stdio.h>
#define N 4

int main(int argc, char** argv) {
    int rank, size, i, j, k;
    int A[N][N], B[N][N], C[N][N];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Initialize A, B, and C, distribute rows, perform multiplication, gather results
    // ... (implementation details)
    MPI_Finalize();
    return 0;
}
```

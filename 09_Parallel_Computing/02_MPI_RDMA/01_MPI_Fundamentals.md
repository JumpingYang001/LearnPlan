# 1. MPI Fundamentals

## Description
Covers the basics of MPI, including its history, standards, programming model, point-to-point communication, and environment management.

## Example Code
```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    printf("Hello from rank %d\n", world_rank);
    MPI_Finalize();
    return 0;
}
```

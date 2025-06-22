# 3. MPI Collective Communication

## Description
Covers collective operations such as barriers, broadcast, scatter/gather, reductions, and their non-blocking and optimized variants.

## Example Code
```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int data = rank;
    int sum = 0;
    MPI_Reduce(&data, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Sum = %d\n", sum);
    }
    MPI_Finalize();
    return 0;
}
```

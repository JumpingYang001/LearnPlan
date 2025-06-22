# 2. Advanced MPI Point-to-Point Communication

## Description
Explores non-blocking, persistent, and probe-based messaging, as well as communication modes and their performance implications.

## Example Code
```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int data = 0;
    MPI_Request request;
    if (rank == 0) {
        data = 42;
        MPI_Isend(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &request);
    } else if (rank == 1) {
        MPI_Irecv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        printf("Received %d\n", data);
    }
    MPI_Finalize();
    return 0;
}
```

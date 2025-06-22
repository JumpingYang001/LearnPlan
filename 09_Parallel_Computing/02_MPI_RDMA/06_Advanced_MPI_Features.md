# 6. Advanced MPI Features

## Description
Covers one-sided communication (RMA), MPI I/O, dynamic process management, and thread support.

## Example Code
```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int value = 0;
    MPI_Win win;
    MPI_Win_create(&value, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);
    // ... RMA operations ...
    MPI_Win_fence(0, win);
    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}
```

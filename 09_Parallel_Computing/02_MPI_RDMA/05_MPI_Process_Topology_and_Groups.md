# 5. MPI Process Topology and Groups

## Description
Discusses communicator and group management, Cartesian and graph topologies, and rank translation.

## Example Code
```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int dims[2] = {2, 2};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    // ... use cart_comm ...
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
```

# Hybrid Parallelism

## Description
Understand OpenMP with MPI hybrid programming, process/thread affinity, communication, and hybrid parallel applications.

## Example
```c
// Example: Hybrid OpenMP + MPI (pseudo-code)
#include <mpi.h>
#include <omp.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    #pragma omp parallel
    {
        printf("MPI rank/thread: ...\n");
    }
    MPI_Finalize();
    return 0;
}
```

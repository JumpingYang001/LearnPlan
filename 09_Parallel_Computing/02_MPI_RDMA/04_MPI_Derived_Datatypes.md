# 4. MPI Derived Datatypes

## Description
Explains how to construct and use custom MPI datatypes for efficient communication of complex data structures.

## Example Code
```c
#include <mpi.h>
#include <stdio.h>
typedef struct {
    int a;
    double b;
} mytype;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Datatype mpi_mytype;
    int blocklengths[2] = {1, 1};
    MPI_Aint offsets[2];
    offsets[0] = offsetof(mytype, a);
    offsets[1] = offsetof(mytype, b);
    MPI_Datatype types[2] = {MPI_INT, MPI_DOUBLE};
    MPI_Type_create_struct(2, blocklengths, offsets, types, &mpi_mytype);
    MPI_Type_commit(&mpi_mytype);
    // ... use mpi_mytype ...
    MPI_Type_free(&mpi_mytype);
    MPI_Finalize();
    return 0;
}
```

# OpenMP Basics

## Description
Master OpenMP programming and execution model, compiler directives, thread creation, and basic parallel applications.

## Example
```c
// Example: Parallel region in OpenMP
#include <omp.h>
#include <stdio.h>

int main() {
    #pragma omp parallel
    {
        printf("Hello from thread %d\n", omp_get_thread_num());
    }
    return 0;
}
```

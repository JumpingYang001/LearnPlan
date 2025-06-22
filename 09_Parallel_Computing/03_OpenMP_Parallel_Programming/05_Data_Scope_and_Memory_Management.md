# Data Scope and Memory Management

## Description
Understand shared, private, firstprivate, threadprivate, data dependencies, and race conditions.

## Example
```c
// Example: Data scope in OpenMP
#include <omp.h>
#include <stdio.h>

int main() {
    int x = 10;
    #pragma omp parallel private(x)
    {
        x = omp_get_thread_num();
        printf("Thread %d: x = %d\n", omp_get_thread_num(), x);
    }
    return 0;
}
```

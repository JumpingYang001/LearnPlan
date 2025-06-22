# Parallel Programming Fundamentals

## Description
Understand parallel computing concepts and models, shared vs. distributed memory, parallelism, concurrency, synchronization, performance metrics, and Amdahl's Law.

## Example
```c
// Example: Simple parallel for loop in C with OpenMP
#include <omp.h>
#include <stdio.h>

int main() {
    int i;
    #pragma omp parallel for
    for (i = 0; i < 10; i++) {
        printf("Thread %d processes i = %d\n", omp_get_thread_num(), i);
    }
    return 0;
}
```

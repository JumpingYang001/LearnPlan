# Synchronization

## Description
Master critical sections, atomic operations, barriers, ordered constructs, locks, and mutexes.

## Example
```c
// Example: Synchronization with critical section
#include <omp.h>
#include <stdio.h>

int main() {
    int sum = 0;
    #pragma omp parallel for
    for (int i = 0; i < 100; i++) {
        #pragma omp critical
        sum += i;
    }
    printf("Sum = %d\n", sum);
    return 0;
}
```

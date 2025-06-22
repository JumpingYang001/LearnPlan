# Work Sharing Constructs

## Description
Understand for, sections, single directives, scheduling, loop parallelization, and work sharing.

## Example
```c
// Example: Work sharing with for directive
#include <omp.h>
#include <stdio.h>

int main() {
    int i;
    #pragma omp parallel for schedule(dynamic)
    for (i = 0; i < 8; i++) {
        printf("Thread %d handles i = %d\n", omp_get_thread_num(), i);
    }
    return 0;
}
```

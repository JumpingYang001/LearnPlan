# Timing Analysis and Verification

## Description
Covers worst-case execution time (WCET) analysis, static timing analysis, schedulability analysis, and timing verification for real-time systems.

## Example Code: WCET Measurement (C)
```c
#include <stdio.h>
#include <time.h>

void task() {
    // Simulate work
    for (volatile int i = 0; i < 1000000; ++i);
}

int main() {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    task();
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("WCET: %ld ns\n", (end.tv_nsec - start.tv_nsec));
    return 0;
}
```

# Real-Time Systems Fundamentals

## Description
Covers the basic characteristics, requirements, and types of real-time systems. Explains determinism, predictability, and responsiveness, and contrasts real-time with general-purpose systems.

## Example Code: Simple Real-Time Task (C)
```c
#include <stdio.h>
#include <time.h>

void real_time_task() {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // Simulate work
    for (volatile int i = 0; i < 1000000; ++i);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Task duration: %ld ns\n", (end.tv_nsec - start.tv_nsec));
}

int main() {
    real_time_task();
    return 0;
}
```

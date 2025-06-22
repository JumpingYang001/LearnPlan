# Advanced Optimization Topics

Discusses benchmarking, performance budgeting, hardware acceleration, and profile-driven development.

## Microbenchmark Example (C)
```c
#include <stdio.h>
#include <time.h>
void bench() {
    // code to benchmark
}
int main() {
    clock_t start = clock();
    bench();
    clock_t end = clock();
    printf("Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
}
```

## Profile-Driven Optimization Example (C++)
```cpp
__attribute__((hot)) void hot_func() { /* ... */ }
__attribute__((cold)) void cold_func() { /* ... */ }
```

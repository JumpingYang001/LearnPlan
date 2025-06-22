# Project: Benchmark Suite Development

Design comprehensive benchmarks for a library or system and create automated performance regression testing.

## Example: Benchmarking a Math Library in C
```c
#include <stdio.h>
#include <math.h>
#include <time.h>
int main() {
    double sum = 0;
    clock_t start = clock();
    for (int i = 0; i < 10000000; ++i) {
        sum += sin(i) + cos(i);
    }
    clock_t end = clock();
    printf("Sum: %f, Time: %f s\n", sum, (double)(end - start) / CLOCKS_PER_SEC);
}
```

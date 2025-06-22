# Performance Profiling Fundamentals

This section introduces the core concepts of performance profiling, including CPU and memory profiling, sampling vs. instrumentation, and key performance metrics.

## Profiling Concepts
- **CPU profiling vs. memory profiling**: CPU profiling measures where CPU time is spent, while memory profiling tracks memory usage and leaks.
- **Sampling vs. instrumentation**: Sampling periodically checks the program counter, while instrumentation inserts code to record events.
- **Call graphs and hot paths**: Visualize function call relationships and identify performance-critical paths.

### Example: Simple CPU Profiler in C
```c
#include <stdio.h>
#include <time.h>

void work() {
    for (volatile int i = 0; i < 100000000; ++i);
}

int main() {
    clock_t start = clock();
    work();
    clock_t end = clock();
    printf("Elapsed: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    return 0;
}
```

## Performance Metrics
- **Latency, throughput, utilization**: Key metrics for evaluating performance.
- **Cache hit/miss rates, branch prediction, instruction cycles**: Hardware-level metrics for deeper analysis.

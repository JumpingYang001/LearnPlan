# Advanced OpenMP Features

## Description
Master OpenMP 5.0/5.1 features, device constructs, target directives, offloading, and advanced OpenMP applications.

## Example
```c
// Example: OpenMP target offloading (requires OpenMP 4.5+)
#include <omp.h>
#include <stdio.h>

int main() {
    int a = 10, b = 20, c = 0;
    #pragma omp target map(to:a,b) map(from:c)
    {
        c = a + b;
    }
    printf("Result on host: %d\n", c);
    return 0;
}
```

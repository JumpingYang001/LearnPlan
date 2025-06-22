# Loop Optimizations

## Overview
Explains loop unrolling, fusion, loop-invariant code motion, tiling, and interchange.

## Example: Loop Unrolling and Invariant Code Motion (C/C++)
```c
void sum(int *a, int n) {
    int s = 0;
    for (int i = 0; i < n; ++i) {
        s += a[i];
    }
    // Loop unrolling example
    for (int i = 0; i < n; i += 4) {
        s += a[i] + a[i+1] + a[i+2] + a[i+3];
    }
}
```
// Use -funroll-loops or -O3 to enable loop unrolling.

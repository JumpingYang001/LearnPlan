# Project: Auto-Vectorization Helper

## Description
Develop a tool to assist in code vectorization, implement code transformations, and analyze vectorization success.

## Example (C/C++)
```c
// Example: Vectorizable loop
void add_arrays(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
```
// Compile with -O2/-O3 and -ftree-vectorize. Analyze with compiler reports.

# Memory Hierarchy Optimizations

## Overview
Discusses cache-conscious programming, prefetching, and data layout.

## Example: Cache-Friendly Access (C/C++)
```c
void sum_rows(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        int sum = 0;
        for (int j = 0; j < cols; ++j) {
            sum += matrix[i][j];
        }
    }
}
```
// Accessing memory row-wise improves cache locality.

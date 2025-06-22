# Memory Management Fundamentals

## Overview
This section covers the basics of memory allocation, types of memory issues, and management strategies in C/C++.

### Memory Allocation in C/C++
```c
#include <stdio.h>
#include <stdlib.h>

int main() {
    int *arr = (int*)malloc(5 * sizeof(int)); // heap allocation
    for(int i = 0; i < 5; i++) arr[i] = i;
    free(arr); // always free allocated memory
    return 0;
}
```

### Common Issues
- Memory leaks
- Double free
- Use-after-free
- Buffer overflows/underflows

### RAII and Smart Pointers (C++)
```cpp
#include <memory>
void foo() {
    std::unique_ptr<int[]> arr(new int[5]);
    // arr is automatically freed
}
```

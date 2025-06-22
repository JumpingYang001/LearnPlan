# Project 1: Custom Memory Leak Detector

## Description
Implement a simple memory tracking library in C/C++ and create visualization tools for memory usage.

### Example: Simple Memory Tracker
```c
#include <stdio.h>
#include <stdlib.h>
static size_t total_alloc = 0;
void* my_malloc(size_t size) {
    total_alloc += size;
    return malloc(size);
}
void my_free(void* ptr, size_t size) {
    total_alloc -= size;
    free(ptr);
}
int main() {
    int* arr = (int*)my_malloc(10 * sizeof(int));
    my_free(arr, 10 * sizeof(int));
    printf("Total allocated: %zu\n", total_alloc);
    return 0;
}
```

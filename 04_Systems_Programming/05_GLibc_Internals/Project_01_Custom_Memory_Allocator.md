# Project: Custom Memory Allocator

## Description
Implement a specialized memory allocator for specific workloads, compare its performance with standard malloc, add debugging/profiling features, and optimize for different use cases.

## Example: Simple Custom Allocator
```c
#include <stdlib.h>
#include <stdio.h>

void* my_malloc(size_t size) {
    void* ptr = malloc(size);
    printf("[my_malloc] Allocated %zu bytes at %p\n", size, ptr);
    return ptr;
}

void my_free(void* ptr) {
    printf("[my_free] Freeing memory at %p\n", ptr);
    free(ptr);
}

int main() {
    int* arr = (int*)my_malloc(100 * sizeof(int));
    my_free(arr);
    return 0;
}
```

# Memory Management in glibc

## Description
Explains glibc's malloc implementation, memory allocation algorithms (ptmalloc2), debugging tools, and custom allocators.

## Example: Custom Allocator
```c
#include <stdlib.h>
#include <stdio.h>

void* my_malloc(size_t size) {
    void* ptr = malloc(size);
    printf("Allocated %zu bytes at %p\n", size, ptr);
    return ptr;
}

int main() {
    int* arr = (int*)my_malloc(10 * sizeof(int));
    free(arr);
    return 0;
}
```

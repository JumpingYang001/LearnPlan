# Project: Library Function Interposer

## Description
Develop a framework for intercepting glibc function calls, implement monitoring/debugging, add performance enhancements, and demonstrate use cases for security and profiling.

## Example: Interposing malloc
```c
#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>

void* malloc(size_t size) {
    static void* (*real_malloc)(size_t) = NULL;
    if (!real_malloc) real_malloc = dlsym(RTLD_NEXT, "malloc");
    void* ptr = real_malloc(size);
    printf("[interposed malloc] %zu bytes at %p\n", size, ptr);
    return ptr;
}
```

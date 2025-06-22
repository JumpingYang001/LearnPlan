# Memory Management

## Description
This section discusses virtual memory, allocation, and memory-mapped files in Windows. It includes a C example of allocating and freeing memory.

## Example: Virtual Memory Allocation (C)

```c
#include <windows.h>
#include <stdio.h>

int main() {
    SIZE_T size = 1024 * 1024; // 1 MB
    LPVOID lpMem = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (lpMem) {
        printf("Memory allocated!\n");
        VirtualFree(lpMem, 0, MEM_RELEASE);
    } else {
        printf("Memory allocation failed.\n");
    }
    return 0;
}
```

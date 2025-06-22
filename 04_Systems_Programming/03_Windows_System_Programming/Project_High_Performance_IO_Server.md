# Project: High-Performance I/O Server

## Description
Build a server application using I/O completion ports, efficient thread pooling, and performance benchmarking tools.

## Example: Create I/O Completion Port (C)

```c
#include <windows.h>
#include <stdio.h>

int main() {
    HANDLE hIOCP = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, 0);
    if (hIOCP) {
        printf("I/O Completion Port created!\n");
        CloseHandle(hIOCP);
    } else {
        printf("Failed to create IOCP.\n");
    }
    return 0;
}
```

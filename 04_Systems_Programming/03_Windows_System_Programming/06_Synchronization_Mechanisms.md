# Synchronization Mechanisms

## Description
This section explains synchronization primitives like critical sections, mutexes, and semaphores in Windows. Below is a C example using a mutex.

## Example: Mutex Synchronization (C)

```c
#include <windows.h>
#include <stdio.h>

int main() {
    HANDLE hMutex = CreateMutex(NULL, FALSE, NULL);
    if (hMutex) {
        WaitForSingleObject(hMutex, INFINITE);
        printf("In critical section\n");
        ReleaseMutex(hMutex);
        CloseHandle(hMutex);
    }
    return 0;
}
```

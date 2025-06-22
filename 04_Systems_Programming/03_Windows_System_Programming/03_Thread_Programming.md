# Thread Programming

## Description
This section explains thread creation, management, and synchronization in Windows. It provides a C example of creating a thread.

## Example: Create a Thread (C)

```c
#include <windows.h>
#include <stdio.h>

DWORD WINAPI ThreadFunc(LPVOID lpParam) {
    printf("Hello from thread!\n");
    return 0;
}

int main() {
    HANDLE hThread = CreateThread(NULL, 0, ThreadFunc, NULL, 0, NULL);
    if (hThread) {
        WaitForSingleObject(hThread, INFINITE);
        CloseHandle(hThread);
    }
    return 0;
}
```

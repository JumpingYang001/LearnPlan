# Process Management

## Description
This section covers process creation, termination, and management using the Windows API. It demonstrates how to create a new process in C.

## Example: Create a Process (C)

```c
#include <windows.h>
#include <stdio.h>

int main() {
    STARTUPINFO si = {0};
    PROCESS_INFORMATION pi = {0};
    si.cb = sizeof(si);
    if (CreateProcess(NULL, "notepad.exe", NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        printf("Process created!\n");
        WaitForSingleObject(pi.hProcess, INFINITE);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    } else {
        printf("Failed to create process.\n");
    }
    return 0;
}
```

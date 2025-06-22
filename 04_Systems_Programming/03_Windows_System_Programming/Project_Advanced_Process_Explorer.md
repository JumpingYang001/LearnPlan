# Project: Advanced Process Explorer

## Description
Build a utility for monitoring and controlling processes, displaying detailed process and thread information, and providing process/memory manipulation tools.

## Example: List Running Processes (C)

```c
#include <windows.h>
#include <tlhelp32.h>
#include <stdio.h>

int main() {
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    PROCESSENTRY32 pe = {0};
    pe.dwSize = sizeof(pe);
    if (Process32First(hSnapshot, &pe)) {
        do {
            printf("Process: %s (PID: %lu)\n", pe.szExeFile, pe.th32ProcessID);
        } while (Process32Next(hSnapshot, &pe));
    }
    CloseHandle(hSnapshot);
    return 0;
}
```

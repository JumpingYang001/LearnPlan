# Chrome Process Architecture

## Overview
Chrome uses a multi-process architecture to improve security, stability, and performance. Each tab, extension, and plugin may run in its own process, isolated from others.

## Key Concepts
- Browser, renderer, plugin, and utility processes
- Inter-process communication (IPC)
- Task scheduling system

## Example: Simple Process Creation in C
```c
#include <stdio.h>
#include <windows.h>

int main() {
    STARTUPINFO si = { sizeof(si) };
    PROCESS_INFORMATION pi;
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

This example demonstrates basic process creation, similar to how Chrome launches isolated processes for tabs and extensions.

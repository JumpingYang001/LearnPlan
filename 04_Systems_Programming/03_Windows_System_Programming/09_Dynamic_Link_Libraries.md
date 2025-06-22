# Dynamic Link Libraries (DLLs)

## Description
This section explains DLL creation, dynamic loading, and entry points in Windows. Below is a C example of a simple DLL.

## Example: Simple DLL (C)

```c
// mydll.c
#include <windows.h>

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

__declspec(dllexport) int add(int a, int b) {
    return a + b;
}
```

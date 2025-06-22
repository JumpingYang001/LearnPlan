# Process Management in Windows

## Description
Windows process/thread objects, EPROCESS/ETHREAD, scheduling, and comparison with Linux.

## C++ Example: CreateProcess
```cpp
#include <windows.h>
#include <iostream>

int main() {
    STARTUPINFO si = { sizeof(si) };
    PROCESS_INFORMATION pi;
    if (CreateProcess(NULL, (LPSTR)"notepad.exe", NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        std::cout << "Process created!" << std::endl;
        WaitForSingleObject(pi.hProcess, INFINITE);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }
    return 0;
}
```

# File Systems in Windows

## Description
NTFS architecture, ReFS, cache manager, memory-mapped files, comparison with Linux.

## C++ Example: CreateFile and WriteFile
```cpp
#include <windows.h>
#include <iostream>

int main() {
    HANDLE hFile = CreateFile("test.txt", GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile != INVALID_HANDLE_VALUE) {
        DWORD written;
        WriteFile(hFile, "Hello, file!", 11, &written, NULL);
        CloseHandle(hFile);
    }
    return 0;
}
```

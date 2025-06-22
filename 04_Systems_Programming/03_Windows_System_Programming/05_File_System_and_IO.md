# File System and I/O

## Description
This section covers file and directory operations, synchronous/asynchronous I/O, and file attributes in Windows. Below is a C example for basic file I/O.

## Example: File Write and Read (C)

```c
#include <windows.h>
#include <stdio.h>

int main() {
    HANDLE hFile = CreateFile("test.txt", GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile != INVALID_HANDLE_VALUE) {
        DWORD written;
        WriteFile(hFile, "Hello, File!", 12, &written, NULL);
        CloseHandle(hFile);
    }
    hFile = CreateFile("test.txt", GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile != INVALID_HANDLE_VALUE) {
        char buffer[20] = {0};
        DWORD read;
        ReadFile(hFile, buffer, 12, &read, NULL);
        printf("Read: %s\n", buffer);
        CloseHandle(hFile);
    }
    return 0;
}
```

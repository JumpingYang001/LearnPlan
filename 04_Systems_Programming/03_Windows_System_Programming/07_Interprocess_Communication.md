# Interprocess Communication

## Description
This section covers pipes, mailslots, file mapping, and Windows messaging. Below is a C example of an anonymous pipe.

## Example: Anonymous Pipe (C)

```c
#include <windows.h>
#include <stdio.h>

int main() {
    HANDLE hRead, hWrite;
    char buffer[100];
    DWORD written, read;
    if (CreatePipe(&hRead, &hWrite, NULL, 0)) {
        WriteFile(hWrite, "Pipe Message", 12, &written, NULL);
        ReadFile(hRead, buffer, 12, &read, NULL);
        buffer[read] = '\0';
        printf("Received: %s\n", buffer);
        CloseHandle(hRead);
        CloseHandle(hWrite);
    }
    return 0;
}
```

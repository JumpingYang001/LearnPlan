# Project: Inter-Process Communication Framework

## Description
Build a framework supporting multiple IPC mechanisms, transparent serialization, message routing, and secure communication channels.

## Example: Named Pipe Server (C)

```c
#include <windows.h>
#include <stdio.h>

int main() {
    HANDLE hPipe = CreateNamedPipe(
        "\\.\pipe\MyPipe", PIPE_ACCESS_DUPLEX, PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
        1, 1024, 1024, 0, NULL);
    if (hPipe == INVALID_HANDLE_VALUE) {
        printf("Failed to create named pipe.\n");
        return 1;
    }
    printf("Waiting for client...\n");
    ConnectNamedPipe(hPipe, NULL);
    char buffer[128];
    DWORD bytesRead;
    ReadFile(hPipe, buffer, sizeof(buffer)-1, &bytesRead, NULL);
    buffer[bytesRead] = '\0';
    printf("Received: %s\n", buffer);
    CloseHandle(hPipe);
    return 0;
}
```

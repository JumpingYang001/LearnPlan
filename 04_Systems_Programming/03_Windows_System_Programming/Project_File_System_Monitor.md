# Project: File System Monitor

## Description
Develop an application that monitors file system changes and provides real-time notification of file operations.

## Example: Monitor Directory Changes (C)

```c
#include <windows.h>
#include <stdio.h>

int main() {
    HANDLE hDir = CreateFile(
        ".", FILE_LIST_DIRECTORY, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
    if (hDir == INVALID_HANDLE_VALUE) return 1;
    char buffer[1024];
    DWORD bytesReturned;
    while (ReadDirectoryChangesW(hDir, buffer, sizeof(buffer), TRUE,
        FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_DIR_NAME | FILE_NOTIFY_CHANGE_LAST_WRITE,
        &bytesReturned, NULL, NULL)) {
        printf("Directory changed!\n");
    }
    CloseHandle(hDir);
    return 0;
}
```

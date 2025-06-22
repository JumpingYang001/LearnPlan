# Advanced I/O and Device Access

## Description
This section discusses device I/O control, overlapped I/O, and completion routines. Below is a C example for DeviceIoControl usage.

## Example: DeviceIoControl (C)

```c
#include <windows.h>
#include <stdio.h>

int main() {
    HANDLE hDevice = CreateFile("\\\\.\\PhysicalDrive0", GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    if (hDevice != INVALID_HANDLE_VALUE) {
        DWORD bytesReturned;
        DeviceIoControl(hDevice, IOCTL_STORAGE_CHECK_VERIFY, NULL, 0, NULL, 0, &bytesReturned, NULL);
        CloseHandle(hDevice);
    } else {
        printf("Failed to open device.\n");
    }
    return 0;
}
```

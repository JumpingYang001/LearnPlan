# Windows Security Programming

## Description
This section covers security descriptors, access control, privileges, and impersonation in Windows. Below is a C example for checking process token privileges.

## Example: Check Token Privileges (C)

```c
#include <windows.h>
#include <stdio.h>

int main() {
    HANDLE hToken;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &hToken)) {
        TOKEN_PRIVILEGES tp;
        DWORD len = sizeof(tp);
        if (GetTokenInformation(hToken, TokenPrivileges, &tp, sizeof(tp), &len)) {
            printf("Privileges count: %lu\n", tp.PrivilegeCount);
        }
        CloseHandle(hToken);
    } else {
        printf("Failed to open process token.\n");
    }
    return 0;
}
```

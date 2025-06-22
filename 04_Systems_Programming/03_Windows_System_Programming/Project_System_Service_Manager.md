# Project: System Service Manager

## Description
Develop a comprehensive service management application for installing, configuring, and controlling Windows services.

## Example: Query Service Status (C)

```c
#include <windows.h>
#include <stdio.h>

int main() {
    SC_HANDLE scm = OpenSCManager(NULL, NULL, SC_MANAGER_ALL_ACCESS);
    if (!scm) return 1;
    SC_HANDLE svc = OpenService(scm, "wuauserv", SERVICE_QUERY_STATUS);
    if (svc) {
        SERVICE_STATUS status;
        if (QueryServiceStatus(svc, &status)) {
            printf("Service status: %d\n", status.dwCurrentState);
        }
        CloseServiceHandle(svc);
    }
    CloseServiceHandle(scm);
    return 0;
}
```

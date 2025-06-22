# Device Drivers in Windows

## Description
Windows driver models (WDM, KMDF, UMDF), driver objects, IRPs, PnP, power management, comparison with Linux.

## C++ Example: KMDF Driver Skeleton
```cpp
#include <ntddk.h>

extern "C" NTSTATUS DriverEntry(PDRIVER_OBJECT DriverObject, PUNICODE_STRING RegistryPath) {
    UNREFERENCED_PARAMETER(DriverObject);
    UNREFERENCED_PARAMETER(RegistryPath);
    DbgPrint("KMDF Driver Loaded\n");
    return STATUS_SUCCESS;
}
```

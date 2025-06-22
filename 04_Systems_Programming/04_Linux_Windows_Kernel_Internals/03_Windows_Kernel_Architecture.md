# Windows Kernel Architecture

## Description
Windows NT kernel, Executive, HAL, kernel components, and subsystems. Comparison with Linux.

## C++ Example: Windows Driver Entry
```cpp
#include <ntddk.h>

extern "C" NTSTATUS DriverEntry(PDRIVER_OBJECT DriverObject, PUNICODE_STRING RegistryPath) {
    UNREFERENCED_PARAMETER(DriverObject);
    UNREFERENCED_PARAMETER(RegistryPath);
    DbgPrint("Hello, Windows kernel!\n");
    return STATUS_SUCCESS;
}
```

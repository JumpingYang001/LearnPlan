# Project: Cross-Platform Kernel Module

## Description
Build modules/drivers that work on both Windows and Linux. Implement consistent functionality and abstraction layers for kernel differences.

## Linux Example: Simple Kernel Module
```c
#include <linux/module.h>
#include <linux/kernel.h>

int init_module(void) {
    printk(KERN_INFO "Linux module loaded\n");
    return 0;
}
void cleanup_module(void) {
    printk(KERN_INFO "Linux module unloaded\n");
}
```

## Windows Example: Driver Skeleton
```cpp
#include <ntddk.h>
extern "C" NTSTATUS DriverEntry(PDRIVER_OBJECT DriverObject, PUNICODE_STRING RegistryPath) {
    DbgPrint("Windows driver loaded\n");
    return STATUS_SUCCESS;
}
```

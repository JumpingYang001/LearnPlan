# Kernel Debugging and Tracing

## Description
Linux kernel debugging, Windows kernel debugging, tracing (ftrace, eBPF, ETW), implementation.

## C Example: Using ftrace (Linux)
```c
// Enable function tracing in Linux kernel
// echo function > /sys/kernel/debug/tracing/current_tracer
// cat /sys/kernel/debug/tracing/trace
```

## C++ Example: DbgPrint (Windows)
```cpp
#include <ntddk.h>

extern "C" void DebugTrace() {
    DbgPrint("Tracing in Windows kernel\n");
}
```

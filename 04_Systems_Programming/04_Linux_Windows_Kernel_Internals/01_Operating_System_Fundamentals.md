# Operating System Fundamentals

## Description
Overview of OS concepts, kernel vs. user mode, system calls, and protection rings. Comparison of Linux and Windows architectures.

## C Example: System Call (Linux)
```c
#include <stdio.h>
#include <unistd.h>

int main() {
    printf("PID: %d\n", getpid());
    return 0;
}
```

## C++ Example: Windows System Call
```cpp
#include <windows.h>
#include <iostream>

int main() {
    std::cout << "Process ID: " << GetCurrentProcessId() << std::endl;
    return 0;
}
```

# Project: Compiler Performance Analysis

## Goal
Analyze and optimize compilation time. Create tools for compiler performance profiling.

## Example: Timing Compilation (C)
```c
// Use time command to measure compilation
time gcc -O2 -c large_file.c
```

## Example: Custom Profiling Tool (C++)
```cpp
#include <chrono>
#include <iostream>
int main() {
    auto start = std::chrono::high_resolution_clock::now();
    // Compilation or code to profile
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    return 0;
}
```

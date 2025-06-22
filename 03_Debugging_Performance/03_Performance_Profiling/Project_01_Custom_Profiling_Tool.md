# Project: Custom Profiling Tool

Create a specialized profiler for a specific use case and visualize performance data.

## Example: Simple Function Timer in C++
```cpp
#include <chrono>
#include <iostream>
void profiled_function() {
    // Simulate work
    for (volatile int i = 0; i < 10000000; ++i);
}
int main() {
    auto start = std::chrono::high_resolution_clock::now();
    profiled_function();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed: " << std::chrono::duration<double>(end - start).count() << " s\n";
}
```

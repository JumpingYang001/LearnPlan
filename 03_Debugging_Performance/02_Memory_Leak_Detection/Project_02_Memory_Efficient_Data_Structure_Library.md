# Project 2: Memory-Efficient Data Structure Library

## Description
Design and implement cache-friendly containers and benchmark against standard containers.

### Example: Cache-Friendly Vector
```cpp
#include <vector>
#include <chrono>
#include <iostream>
int main() {
    std::vector<int> v(1000000, 1);
    auto start = std::chrono::high_resolution_clock::now();
    long long sum = 0;
    for (auto x : v) sum += x;
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Sum: " << sum << "\n";
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us\n";
}
```

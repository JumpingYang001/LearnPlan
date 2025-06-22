# Project: Chrome Extension with Performance Analysis

## Description
Build a Chrome extension that analyzes page performance, visualizes rendering metrics, and provides optimization recommendations. Add support for comparing multiple pages.

## Example: Simulated Performance Data Collection in C++
```cpp
#include <iostream>
#include <vector>

struct PageMetrics {
    double renderTime;
    double scriptTime;
};

void analyze(const std::vector<PageMetrics>& pages) {
    for (size_t i = 0; i < pages.size(); ++i) {
        std::cout << "Page " << i+1 << ": Render " << pages[i].renderTime << "ms, Script " << pages[i].scriptTime << "ms\n";
    }
}

int main() {
    std::vector<PageMetrics> data = {{120.5, 30.2}, {98.7, 25.1}};
    analyze(data);
    return 0;
}
```

This code simulates collecting and comparing performance metrics for multiple pages.

# Project: TCP Congestion Control Visualization

## Objective
Build a tool to visualize TCP congestion control algorithms. Show window size changes during transmission.

## Example Code (Python, matplotlib)
```python
## Example Code (C++)
```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> window_sizes = {1, 2, 4, 8, 16, 12, 8, 4};
    std::cout << "# Time WindowSize\n";
    for (size_t i = 0; i < window_sizes.size(); ++i) {
        std::cout << i << " " << window_sizes[i] << std::endl;
    }
    std::cout << "\n# Use gnuplot or another tool to plot the data above." << std::endl;
    return 0;
}
```
This C++ program outputs time vs. window size data. You can redirect the output to a file and plot it using gnuplot or any plotting tool:

```
g++ -o tcp_window tcp_window.cpp
./tcp_window > window_data.txt
gnuplot -e "plot 'window_data.txt' using 1:2 with linespoints title 'TCP Congestion Window'"
```
```

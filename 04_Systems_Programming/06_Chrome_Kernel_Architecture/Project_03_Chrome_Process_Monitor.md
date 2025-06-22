# Project: Chrome Process Monitor

## Description
Build a tool to visualize Chrome's process model, track resource usage, and visualize inter-process communication. Add anomaly detection and alerting.

## Example: Process Resource Tracking in C++
```cpp
#include <iostream>
#include <vector>

struct ProcessInfo {
    int pid;
    double cpuUsage;
    double memUsage;
};

void printProcesses(const std::vector<ProcessInfo>& procs) {
    for (const auto& p : procs) {
        std::cout << "PID: " << p.pid << ", CPU: " << p.cpuUsage << "% , MEM: " << p.memUsage << "MB\n";
    }
}

int main() {
    std::vector<ProcessInfo> procs = {{1234, 2.5, 150.0}, {5678, 1.2, 100.0}};
    printProcesses(procs);
    return 0;
}
```

This code simulates process monitoring, similar to Chrome's process visualization.

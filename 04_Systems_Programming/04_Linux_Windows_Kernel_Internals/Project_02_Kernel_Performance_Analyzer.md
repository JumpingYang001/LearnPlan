# Project: Kernel Performance Analyzer

## Description
Develop a tool to analyze kernel performance metrics and visualize system behavior. Comparative analysis between Linux and Windows.

## Linux Example: Read /proc/stat
```c
#include <stdio.h>
int main() {
    FILE *f = fopen("/proc/stat", "r");
    char buf[256];
    while (fgets(buf, sizeof(buf), f)) {
        printf("%s", buf);
    }
    fclose(f);
    return 0;
}
```

## Windows Example: Query System Information
```cpp
#include <windows.h>
#include <pdh.h>
#include <iostream>
int main() {
    PDH_HQUERY query;
    PDH_HCOUNTER counter;
    PdhOpenQuery(NULL, 0, &query);
    PdhAddCounter(query, L"\\Processor(_Total)\\% Processor Time", 0, &counter);
    PdhCollectQueryData(query);
    std::cout << "Performance data collected." << std::endl;
    PdhCloseQuery(query);
    return 0;
}
```

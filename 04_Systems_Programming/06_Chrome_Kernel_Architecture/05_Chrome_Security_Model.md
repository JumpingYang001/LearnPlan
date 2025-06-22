# Chrome Security Model

## Overview
Chrome uses sandboxing, site isolation, and process separation to provide strong security boundaries between web content and the system.

## Key Concepts
- Sandbox architecture
- Site isolation
- Security mechanisms

## Example: Simple Sandbox Simulation in C++
```cpp
#include <iostream>

void runInSandbox(void (*func)()) {
    std::cout << "Running in sandbox..." << std::endl;
    func();
}

void untrustedCode() {
    std::cout << "Untrusted code executed." << std::endl;
}

int main() {
    runInSandbox(untrustedCode);
    return 0;
}
```

This simulates running code in a sandbox, as Chrome does for web content.

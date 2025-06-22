# LLVM Architecture and Usage

## LLVM Project Overview
- Core components: LLVM Core, Clang, lld, etc.
- Design philosophy and comparison with GCC.

## LLVM Architecture
- LLVM IR, pass infrastructure, target-independent optimization, backends.

**C++ Example:**
```cpp
// Compile with: clang++ -emit-llvm -S example.cpp
#include <iostream>
int main() {
    std::cout << "Hello, LLVM!" << std::endl;
    return 0;
}
```

## Clang Front-end
- Command-line, GCC compatibility, diagnostics, static analysis.

## Other LLVM Tools
- lld, lldb, opt, llvm-objdump, etc.

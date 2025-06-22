# Clang/LLVM Extensions and Internals

## Clang Language Extensions
- Attributes, pragmas, built-ins, OpenCL/CUDA support.

**C++ Example:**
```cpp
#include <iostream>
[[nodiscard]] int compute() { return 42; }
int main() {
    std::cout << compute() << std::endl;
    return 0;
}
```

## LLVM Intrinsics
- Core/platform-specific intrinsics, SIMD/vector ops.

## LibTooling and Clang Tools
- AST traversal, source transformation, analyzers, build system integration.

## Writing LLVM Passes
- Pass types, analysis/transformation, registration, execution.

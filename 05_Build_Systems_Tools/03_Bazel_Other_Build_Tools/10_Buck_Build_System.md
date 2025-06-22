# Buck Build System

## Overview
Learn Buck's concepts, dependency management, and performance optimizations. Compare Buck with Bazel and see sample projects.

## Example: Buck C++ Binary
```python
# BUCK file
cxx_binary(
    name = "hello-buck",
    srcs = ["hello.cpp"],
)
```
```cpp
// hello.cpp
#include <iostream>
int main() {
    std::cout << "Hello from Buck!" << std::endl;
    return 0;
}
```

# Meson Build System

## Overview
Learn Meson's syntax, cross-compilation, and integration with other tools. Compare Meson with other build systems.

## Example: Meson C++ Project
```meson
# meson.build
project('hello', 'cpp')
executable('hello', 'hello.cpp')
```
```cpp
// hello.cpp
#include <iostream>
int main() {
    std::cout << "Hello from Meson!" << std::endl;
    return 0;
}
```

# Modern C++ and CMake

## C++ Standard Selection
- target_compile_features, CMAKE_CXX_STANDARD

### Example: C++ Standard
```cmake
add_executable(myexe main.cpp)
target_compile_features(myexe PRIVATE cxx_std_17)
```

```cpp
// main.cpp
#include <iostream>
int main() {
    std::cout << "Hello Modern C++!" << std::endl;
    return 0;
}
```

# C++ Specific Debugging

## Overview
Debug C++ templates, overloaded functions, STL containers, exceptions, and virtual functions in GDB.

## Example: STL Container Inspection
```cpp
#include <vector>
#include <iostream>
int main() {
    std::vector<int> v = {1,2,3};
    std::cout << v[1] << std::endl;
    return 0;
}
```

GDB commands:
```
break main
run
print v
```

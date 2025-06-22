# Testing with CTest

## CTest Basics
- enable_testing, add_test

### Example: Add Test
```cmake
enable_testing()
add_executable(test_hello test_hello.c)
add_test(NAME HelloTest COMMAND test_hello)
```

```c
// test_hello.c
#include <assert.h>
int main() {
    assert(1 == 1);
    return 0;
}
```

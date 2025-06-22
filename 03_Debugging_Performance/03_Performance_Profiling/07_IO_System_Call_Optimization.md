# I/O and System Call Optimization

Discusses file and network I/O optimization, and system call reduction techniques.

## Buffered File I/O Example (C)
```c
#include <stdio.h>
int main() {
    FILE *f = fopen("data.txt", "r");
    char buf[1024];
    while (fgets(buf, sizeof(buf), f)) {
        // process line
    }
    fclose(f);
    return 0;
}
```

## Asynchronous I/O Example (C++)
```cpp
#include <future>
auto result = std::async(std::launch::async, [](){ /* I/O work */ });
```

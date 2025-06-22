# Project: Multi-Process Application Framework

## Description
Design a framework for creating multi-process applications. Implement process management and communication facilities.

### Example: Simple Multi-Process Manager (C)
```c
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    for (int i = 0; i < 3; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            printf("Child %d running\n", i);
            return 0;
        }
    }
    while (wait(NULL) > 0);
    printf("All children finished\n");
    return 0;
}
```

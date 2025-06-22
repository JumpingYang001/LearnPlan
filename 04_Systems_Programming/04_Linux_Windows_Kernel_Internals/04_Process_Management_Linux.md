# Process Management in Linux

## Description
Linux process representation, scheduling, task structures, context switching, and custom schedulers.

## C Example: Fork and Exec
```c
#include <stdio.h>
#include <unistd.h>

int main() {
    pid_t pid = fork();
    if (pid == 0) {
        execlp("ls", "ls", NULL);
    } else {
        wait(NULL);
    }
    return 0;
}
```

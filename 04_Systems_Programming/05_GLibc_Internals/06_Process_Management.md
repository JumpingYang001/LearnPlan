# Process Management

## Description
Covers process creation (fork, exec), program loading, dynamic linking, and process cleanup in glibc.

## Example: Fork and exec
```c
#include <unistd.h>
#include <stdio.h>

int main() {
    pid_t pid = fork();
    if (pid == 0) {
        execlp("ls", "ls", NULL);
    } else {
        printf("Parent process\n");
    }
    return 0;
}
```

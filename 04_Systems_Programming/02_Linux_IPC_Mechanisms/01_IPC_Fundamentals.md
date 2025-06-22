# IPC Fundamentals

## Overview

Covers the basics of Inter-Process Communication (IPC), including concepts, selection criteria, and Linux process/thread models.

### IPC Concepts
- Process isolation and communication needs
- Local vs. distributed IPC
- Synchronous vs. asynchronous communication
- Message passing vs. shared memory
- Persistence requirements

### Example: Simple Pipe Communication
```c
#include <stdio.h>
#include <unistd.h>
#include <string.h>

int main() {
    int fd[2];
    pipe(fd);
    if (fork() == 0) {
        // Child process
        close(fd[1]);
        char buf[100];
        read(fd[0], buf, sizeof(buf));
        printf("Child received: %s\n", buf);
    } else {
        // Parent process
        close(fd[0]);
        char msg[] = "Hello from parent!";
        write(fd[1], msg, strlen(msg)+1);
    }
    return 0;
}
```

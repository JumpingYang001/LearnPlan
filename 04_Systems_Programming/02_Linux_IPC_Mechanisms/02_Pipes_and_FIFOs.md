# Pipes and FIFOs

## Overview

Explains anonymous pipes and named pipes (FIFOs), their creation, usage, and implementation details.

### Example: Named Pipe (FIFO)
```c
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

int main() {
    const char *fifo = "/tmp/myfifo";
    mkfifo(fifo, 0666);
    if (fork() == 0) {
        // Child: Reader
        int fd = open(fifo, O_RDONLY);
        char buf[100];
        read(fd, buf, sizeof(buf));
        printf("Child read: %s\n", buf);
        close(fd);
    } else {
        // Parent: Writer
        int fd = open(fifo, O_WRONLY);
        char msg[] = "Hello via FIFO!";
        write(fd, msg, strlen(msg)+1);
        close(fd);
    }
    return 0;
}
```

# Memory-Mapped Files

## Overview

Describes memory mapping with mmap, shared memory via mmap, and memory-mapped I/O.

### Example: mmap Shared Memory
```c
#include <stdio.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

int main() {
    int fd = open("/tmp/mmapfile", O_RDWR | O_CREAT, 0666);
    ftruncate(fd, 1024);
    char *data = mmap(NULL, 1024, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    strcpy(data, "Hello mmap!");
    munmap(data, 1024);
    close(fd);
    return 0;
}
```

# Memory Management in Linux

## Description
Linux virtual memory, page tables, TLB, slab allocator, memory zones, OOM killer.

## C Example: mmap Usage
```c
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
    int fd = open("test.txt", O_RDONLY);
    char *data = mmap(NULL, 4096, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data != MAP_FAILED) {
        write(1, data, 4096);
        munmap(data, 4096);
    }
    close(fd);
    return 0;
}
```

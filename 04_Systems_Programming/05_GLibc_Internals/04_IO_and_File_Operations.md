# I/O and File Operations

## Description
Explains file descriptor management, buffered I/O, memory-mapped I/O, and efficient I/O patterns in glibc.

## Example: Buffered and direct I/O
```c
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

int main() {
    FILE* f = fopen("test.txt", "w");
    fprintf(f, "Buffered I/O\n");
    fclose(f);

    int fd = open("test.txt", O_WRONLY | O_APPEND);
    write(fd, "Direct syscall I/O\n", 19);
    close(fd);
    return 0;
}
```

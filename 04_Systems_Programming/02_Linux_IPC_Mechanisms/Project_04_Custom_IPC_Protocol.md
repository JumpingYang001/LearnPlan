# Project: Custom IPC Protocol

## Description
Design and implement a custom IPC protocol for a specific use case. Optimize for performance and reliability.

### Example: Simple Framed Protocol over Pipe (C)
```c
#include <stdio.h>
#include <unistd.h>
#include <string.h>

void send_msg(int fd, const char *msg) {
    uint32_t len = strlen(msg);
    write(fd, &len, sizeof(len));
    write(fd, msg, len);
}

void recv_msg(int fd) {
    uint32_t len;
    read(fd, &len, sizeof(len));
    char buf[256];
    read(fd, buf, len);
    buf[len] = 0;
    printf("Received: %s\n", buf);
}

int main() {
    int fd[2];
    pipe(fd);
    if (fork() == 0) {
        close(fd[1]);
        recv_msg(fd[0]);
    } else {
        close(fd[0]);
        send_msg(fd[1], "Custom protocol message");
    }
    return 0;
}
```

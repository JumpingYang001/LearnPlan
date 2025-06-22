# Socket-Based IPC

## Overview

Explains Unix domain sockets, abstract namespace, and socket options for IPC.

### Example: Unix Domain Socket (Stream)
```c
#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <unistd.h>

int main() {
    int sv[2];
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    if (fork() == 0) {
        close(sv[0]);
        char buf[100];
        read(sv[1], buf, sizeof(buf));
        printf("Child got: %s\n", buf);
    } else {
        close(sv[1]);
        char msg[] = "Socket IPC!";
        write(sv[0], msg, sizeof(msg));
    }
    return 0;
}
```

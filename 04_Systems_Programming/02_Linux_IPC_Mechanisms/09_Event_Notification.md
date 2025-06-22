# Event Notification Mechanisms

## Overview

Describes select, poll, epoll, eventfd, timerfd, and inotify/fanotify.

### Example: epoll Usage
```c
#include <sys/epoll.h>
#include <stdio.h>
#include <unistd.h>

int main() {
    int epfd = epoll_create1(0);
    printf("epoll fd: %d\n", epfd);
    // Add file descriptors and handle events as needed
    close(epfd);
    return 0;
}
```

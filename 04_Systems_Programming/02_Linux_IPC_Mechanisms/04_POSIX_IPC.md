# POSIX IPC

## Overview

Covers POSIX message queues, semaphores, and shared memory, with code examples and comparison to System V IPC.

### Example: POSIX Message Queue
```c
#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <mqueue.h>

int main() {
    mqd_t mq = mq_open("/testqueue", O_CREAT | O_WRONLY, 0644, NULL);
    const char *msg = "Hello POSIX MQ!";
    mq_send(mq, msg, strlen(msg)+1, 0);
    mq_close(mq);
    mq_unlink("/testqueue");
    return 0;
}
```

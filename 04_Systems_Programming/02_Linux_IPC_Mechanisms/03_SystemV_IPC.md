# System V IPC Mechanisms

## Overview

Describes System V message queues, semaphores, and shared memory, including permissions and command-line tools.

### Example: System V Shared Memory
```c
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>

int main() {
    key_t key = ftok("shmfile",65);
    int shmid = shmget(key,1024,0666|IPC_CREAT);
    char *str = (char*) shmat(shmid,(void*)0,0);
    strcpy(str,"Shared memory example");
    printf("Data written: %s\n",str);
    shmdt(str);
    return 0;
}
```

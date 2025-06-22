# Debugging Multi-threaded Programs

## Overview
Debug thread creation, thread-specific breakpoints, deadlock/race detection, and scheduler locking in GDB.

## Example: Thread Breakpoints
```c
#include <pthread.h>
#include <stdio.h>
void* thread_func(void* arg) {
    printf("Thread %d running\n", *(int*)arg);
    return NULL;
}
int main() {
    pthread_t t1, t2;
    int a = 1, b = 2;
    pthread_create(&t1, NULL, thread_func, &a);
    pthread_create(&t2, NULL, thread_func, &b);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    return 0;
}
```

GDB commands:
```
info threads
thread 2
break thread_func if *(int*)arg == 2
```

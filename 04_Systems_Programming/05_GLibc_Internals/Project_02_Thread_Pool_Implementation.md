# Project: Thread Pool Implementation

## Description
Build a thread pool using pthreads, implement a work queue and task scheduling, add monitoring/management features, and optimize for different workloads.

## Example: Minimal Thread Pool
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void* worker(void* arg) {
    printf("Thread %ld working\n", (long)arg);
    return NULL;
}

int main() {
    pthread_t threads[4];
    for (long i = 0; i < 4; ++i)
        pthread_create(&threads[i], NULL, worker, (void*)i);
    for (int i = 0; i < 4; ++i)
        pthread_join(threads[i], NULL);
    return 0;
}
```

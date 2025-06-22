# Threading and Synchronization

## Description
Covers POSIX threads, thread-local storage, synchronization primitives, and thread-safe data structures in glibc.

## Example: Simple pthread usage
```c
#include <pthread.h>
#include <stdio.h>

void* thread_func(void* arg) {
    printf("Hello from thread!\n");
    return NULL;
}

int main() {
    pthread_t tid;
    pthread_create(&tid, NULL, thread_func, NULL);
    pthread_join(tid, NULL);
    return 0;
}
```

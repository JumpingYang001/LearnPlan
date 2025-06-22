# Project: Lock-Free IPC Library

## Description
Create a shared memory IPC library using lock-free techniques. Ensure correct synchronization without mutexes.

### Example: Lock-Free Counter in Shared Memory (C11)
```c
#include <stdatomic.h>
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>

int main() {
    key_t key = ftok("shmfile",65);
    int shmid = shmget(key, sizeof(atomic_int), 0666|IPC_CREAT);
    atomic_int *counter = (atomic_int*) shmat(shmid, (void*)0, 0);
    atomic_store(counter, 0);
    atomic_fetch_add(counter, 1);
    printf("Counter: %d\n", atomic_load(counter));
    shmdt(counter); shmctl(shmid, IPC_RMID, NULL);
    return 0;
}
```

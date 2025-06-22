# Inter-Process Communication in Real-Time Systems

## Description
Explains semaphores, mutexes, message queues, mailboxes, and shared memory for deterministic access. Shows IPC implementation in real-time applications.

## Example Code: Semaphore (POSIX C)
```c
#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>

sem_t sem;

void* task(void* arg) {
    sem_wait(&sem);
    printf("Task running\n");
    sem_post(&sem);
    return NULL;
}

int main() {
    pthread_t t1, t2;
    sem_init(&sem, 0, 1);
    pthread_create(&t1, NULL, task, NULL);
    pthread_create(&t2, NULL, task, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    sem_destroy(&sem);
    return 0;
}
```

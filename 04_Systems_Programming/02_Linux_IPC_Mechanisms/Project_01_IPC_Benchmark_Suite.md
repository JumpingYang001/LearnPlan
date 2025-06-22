# Project: IPC Benchmark Suite

## Description
Implement performance tests for various IPC mechanisms and compare throughput, latency, and resource usage.

### Example: Pipe vs. Shared Memory Benchmark (C)
```c
// This is a simplified benchmark for pipe and shared memory throughput
#include <stdio.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>
#include <time.h>

#define SIZE 4096
#define ITER 10000

void pipe_bench() {
    int fd[2];
    pipe(fd);
    char buf[SIZE] = {0};
    clock_t start = clock();
    for (int i = 0; i < ITER; ++i) {
        write(fd[1], buf, SIZE);
        read(fd[0], buf, SIZE);
    }
    clock_t end = clock();
    printf("Pipe time: %f\n", (double)(end-start)/CLOCKS_PER_SEC);
    close(fd[0]); close(fd[1]);
}

void shm_bench() {
    key_t key = ftok("shmfile",65);
    int shmid = shmget(key, SIZE, 0666|IPC_CREAT);
    char *str = (char*) shmat(shmid,(void*)0,0);
    clock_t start = clock();
    for (int i = 0; i < ITER; ++i) {
        memset(str, 0, SIZE);
    }
    clock_t end = clock();
    printf("Shared memory time: %f\n", (double)(end-start)/CLOCKS_PER_SEC);
    shmdt(str); shmctl(shmid, IPC_RMID, NULL);
}

int main() {
    pipe_bench();
    shm_bench();
    return 0;
}
```

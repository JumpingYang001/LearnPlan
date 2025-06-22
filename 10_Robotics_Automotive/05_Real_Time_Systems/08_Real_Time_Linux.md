# Real-Time Linux

## Description
Covers PREEMPT_RT patch, Xenomai, RTAI, and Linux task scheduling for real-time. Shows how to implement real-time applications on Linux.

## Example Code: Real-Time Scheduling (C)
```c
#include <stdio.h>
#include <sched.h>
#include <unistd.h>

int main() {
    struct sched_param param;
    param.sched_priority = 80;
    if (sched_setscheduler(0, SCHED_FIFO, &param) == -1) {
        perror("sched_setscheduler");
        return 1;
    }
    printf("Real-time scheduling set\n");
    return 0;
}
```

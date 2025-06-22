# Signals

## Overview

Explains standard and real-time signals, signal handling, and signalfd interface.

### Example: Signal Handler
```c
#include <stdio.h>
#include <signal.h>
#include <unistd.h>

void handler(int sig) {
    printf("Caught signal %d\n", sig);
}

int main() {
    signal(SIGUSR1, handler);
    printf("Send SIGUSR1 to %d\n", getpid());
    pause();
    return 0;
}
```

# Signal Handling

## Description
Explains signal delivery, async-signal-safe functions, real-time signals, and robust signal handlers in glibc.

## Example: Signal handler
```c
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

void handler(int sig) {
    write(1, "Signal caught!\n", 15);
}

int main() {
    signal(SIGINT, handler);
    pause();
    return 0;
}
```

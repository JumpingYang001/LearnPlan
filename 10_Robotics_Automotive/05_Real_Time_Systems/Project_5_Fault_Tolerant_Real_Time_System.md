# Project 5: Fault-Tolerant Real-Time System

## Description
Build a system with redundancy and fault detection, implement recovery mechanisms, and create fault injection and testing tools.

## Example Code: Fault Injection (C)
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    srand(time(NULL));
    int data = 100;
    if (rand() % 2 == 0) {
        data = -1; // Inject fault
    }
    if (data < 0) {
        printf("Fault detected!\n");
        // Recovery code here
    } else {
        printf("Data OK: %d\n", data);
    }
    return 0;
}
```

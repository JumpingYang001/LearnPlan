# Fault Tolerance in Real-Time Systems

## Description
Covers redundancy, fault detection, recovery mechanisms, and formal methods for critical systems. Shows how to implement fault-tolerant real-time systems.

## Example Code: Simple Redundancy (C)
```c
#include <stdio.h>

int main() {
    int sensor1 = 100, sensor2 = 102, sensor3 = 101;
    int value = (sensor1 + sensor2 + sensor3) / 3; // Simple voting
    printf("Voted value: %d\n", value);
    return 0;
}
```

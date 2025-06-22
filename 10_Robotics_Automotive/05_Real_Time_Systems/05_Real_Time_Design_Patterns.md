# Real-Time System Design Patterns

## Description
Describes cyclic executive, time-triggered, and event-triggered architectures. Shows implementation of real-time design patterns.

## Example Code: Cyclic Executive (Pseudo-C)
```c
while (1) {
    taskA(); // Every cycle
    if (cycle % 2 == 0) taskB(); // Every 2 cycles
    if (cycle % 5 == 0) taskC(); // Every 5 cycles
    cycle++;
}
```

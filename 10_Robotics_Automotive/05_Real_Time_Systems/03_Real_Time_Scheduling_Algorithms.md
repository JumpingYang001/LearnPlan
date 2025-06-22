# Real-Time Scheduling Algorithms

## Description
Covers rate-monotonic, earliest deadline first (EDF), priority inversion, and priority inheritance. Includes implementation and analysis of scheduling algorithms.

## Example Code: Rate-Monotonic Scheduling (Pseudo-C)
```c
// Pseudo-code for two periodic tasks
while (1) {
    if (time_to_run_task1()) {
        run_task1();
    }
    if (time_to_run_task2()) {
        run_task2();
    }
    // ...
}
```

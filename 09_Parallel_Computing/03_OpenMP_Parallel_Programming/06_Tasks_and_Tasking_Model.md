# Tasks and Tasking Model

## Description
Master the OpenMP task model, task creation, scheduling, dependencies, and task-based parallel applications.

## Example
```c
// Example: OpenMP tasks
#include <omp.h>
#include <stdio.h>

int main() {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < 4; i++) {
                #pragma omp task
                printf("Task %d executed by thread %d\n", i, omp_get_thread_num());
            }
        }
    }
    return 0;
}
```

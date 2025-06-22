# Real-Time Operating Systems (RTOS)

## Description
Explains RTOS architecture, task management, scheduling, and memory management. Compares popular RTOS options like FreeRTOS, VxWorks, and QNX.

## Example Code: FreeRTOS Task Creation (C)
```c
#include "FreeRTOS.h"
#include "task.h"

void vTaskFunction(void *pvParameters) {
    for(;;) {
        // Task code
    }
}

int main(void) {
    xTaskCreate(vTaskFunction, "Task1", 1000, NULL, 1, NULL);
    vTaskStartScheduler();
    for(;;);
}
```

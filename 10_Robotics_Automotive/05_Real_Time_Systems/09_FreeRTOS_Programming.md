# FreeRTOS Programming

## Description
Explains FreeRTOS architecture, API, task creation, management, and synchronization. Shows how to implement applications with FreeRTOS.

## Example Code: FreeRTOS Task (C)
```c
#include "FreeRTOS.h"
#include "task.h"

void vTask(void *pvParameters) {
    for(;;) {
        // Task code
    }
}

int main(void) {
    xTaskCreate(vTask, "Task", 1000, NULL, 1, NULL);
    vTaskStartScheduler();
    for(;;);
}
```

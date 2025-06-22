# Project 3: Embedded RTOS Application

## Description
Build an application on an embedded platform with RTOS, implement resource management and scheduling, and create performance and timing analysis.

## Example Code: FreeRTOS Resource Management (C)
```c
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"

SemaphoreHandle_t xSemaphore;

void vTask(void *pvParameters) {
    for(;;) {
        if (xSemaphoreTake(xSemaphore, portMAX_DELAY)) {
            // Critical section
            xSemaphoreGive(xSemaphore);
        }
    }
}

int main(void) {
    xSemaphore = xSemaphoreCreateMutex();
    xTaskCreate(vTask, "Task", 1000, NULL, 1, NULL);
    vTaskStartScheduler();
    for(;;);
}
```

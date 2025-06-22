# Real-Time Communication Protocols

## Description
Explains real-time communication requirements and protocols such as CAN, FlexRay, and Time-Triggered Ethernet. Shows deterministic networking implementation.

## Example Code: CAN Frame (C)
```c
#include <stdio.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <sys/socket.h>
#include <string.h>

int main() {
    struct can_frame frame;
    memset(&frame, 0, sizeof(frame));
    frame.can_id = 0x123;
    frame.can_dlc = 2;
    frame.data[0] = 0xAB;
    frame.data[1] = 0xCD;
    // Send frame using socketcan (not shown)
    printf("CAN frame prepared\n");
    return 0;
}
```

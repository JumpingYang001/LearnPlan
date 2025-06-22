# Network Access Layer

## Description
Ethernet protocol, ARP, network interface configuration, device drivers, and NIC programming.

## Example
- Ethernet frame format example
- ARP packet structure

### C Example: Ethernet Frame Structure
```c
#include <stdio.h>
#include <stdint.h>

struct ethernet_frame {
    uint8_t dest_mac[6];
    uint8_t src_mac[6];
    uint16_t ethertype;
    uint8_t payload[1500];
};

int main() {
    struct ethernet_frame frame;
    printf("Ethernet frame size: %zu bytes\n", sizeof(frame));
    return 0;
}
```

### C Example: ARP Packet Structure
```c
#include <stdio.h>
#include <stdint.h>

struct arp_packet {
    uint16_t htype;
    uint16_t ptype;
    uint8_t hlen;
    uint8_t plen;
    uint16_t oper;
    uint8_t sha[6];
    uint8_t spa[4];
    uint8_t tha[6];
    uint8_t tpa[4];
};

int main() {
    struct arp_packet arp;
    printf("ARP packet size: %zu bytes\n", sizeof(arp));
    return 0;
}
```

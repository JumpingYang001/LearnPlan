# Internet Layer

## Description
IPv4, IPv6, ICMP, and routing protocols. Includes addressing, subnetting, fragmentation, and protocol details.

## Example
- IPv4 header breakdown
- ICMP echo request/reply example

### C Example: IPv4 Header Structure
```c
#include <stdio.h>
#include <stdint.h>

struct ipv4_header {
    uint8_t ihl:4, version:4;
    uint8_t tos;
    uint16_t tot_len;
    uint16_t id;
    uint16_t frag_off;
    uint8_t ttl;
    uint8_t protocol;
    uint16_t check;
    uint32_t saddr;
    uint32_t daddr;
};

int main() {
    struct ipv4_header ip;
    printf("IPv4 header size: %zu bytes\n", sizeof(ip));
    return 0;
}
```

### C Example: ICMP Echo Request Structure
```c
#include <stdio.h>
#include <stdint.h>

struct icmp_echo {
    uint8_t type;
    uint8_t code;
    uint16_t checksum;
    uint16_t id;
    uint16_t sequence;
};

int main() {
    struct icmp_echo icmp = {8, 0, 0, 1, 1};
    printf("ICMP Echo Request type: %d, code: %d\n", icmp.type, icmp.code);
    return 0;
}
```

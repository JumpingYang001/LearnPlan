# Transport Layer

## Description
TCP and UDP protocols, segment/datagram structure, connection management, flow/congestion control, and ports/sockets.

## Example
- TCP 3-way handshake code
- UDP socket example

### C Example: TCP Header Structure
```c
#include <stdio.h>
#include <stdint.h>

struct tcp_header {
    uint16_t source;
    uint16_t dest;
    uint32_t seq;
    uint32_t ack_seq;
    uint16_t flags;
    uint16_t window;
    uint16_t check;
    uint16_t urg_ptr;
};

int main() {
    struct tcp_header tcp;
    printf("TCP header size: %zu bytes\n", sizeof(tcp));
    return 0;
}
```

### C Example: UDP Header Structure
```c
#include <stdio.h>
#include <stdint.h>

struct udp_header {
    uint16_t source;
    uint16_t dest;
    uint16_t len;
    uint16_t check;
};

int main() {
    struct udp_header udp;
    printf("UDP header size: %zu bytes\n", sizeof(udp));
    return 0;
}
```

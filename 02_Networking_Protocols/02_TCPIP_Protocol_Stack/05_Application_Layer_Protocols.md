# Application Layer Protocols Overview

## Description
Overview of DNS, DHCP, FTP, SMTP, and basic HTTP concepts.

## Example
- DNS query example
- Simple HTTP request

### C Example: DNS Query Packet Structure
```c
#include <stdio.h>
#include <stdint.h>

struct dns_header {
    uint16_t id;
    uint16_t flags;
    uint16_t qdcount;
    uint16_t ancount;
    uint16_t nscount;
    uint16_t arcount;
};

int main() {
    struct dns_header dns;
    printf("DNS header size: %zu bytes\n", sizeof(dns));
    return 0;
}
```

### C Example: Simple HTTP GET Request (string)
```c
#include <stdio.h>

int main() {
    const char *http_get = "GET / HTTP/1.1\r\nHost: example.com\r\n\r\n";
    printf("%s", http_get);
    return 0;
}
```

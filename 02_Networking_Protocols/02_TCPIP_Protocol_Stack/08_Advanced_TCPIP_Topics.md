# Advanced TCP/IP Topics

## Description
TCP performance tuning, IP routing/forwarding, and TCP/IP security considerations.

## Example
- Code for adjusting TCP buffer size
- Example of SYN flood defense

### C Example: Adjusting TCP Buffer Size
```c
#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>

int main() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    int size = 65536;
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size));
    printf("TCP receive buffer set to %d bytes\n", size);
    return 0;
}
```

### C Example: SYN Flood Defense (SYN Cookies)
```c
// SYN cookies are enabled at the OS level, not in user code.
// Example: Enable SYN cookies on Linux
// echo 1 > /proc/sys/net/ipv4/tcp_syncookies
```

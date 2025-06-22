# Network Diagnostics and Analysis

## Description
Diagnostic tools (ping, traceroute, netstat, etc.) and packet analysis (tcpdump, Wireshark).

## Example
- Example tcpdump command
- Wireshark screenshot description

### C Example: Using system() to call ping
```c
#include <stdlib.h>

int main() {
    system("ping -c 4 8.8.8.8");
    return 0;
}
```

### C Example: Using system() to call tcpdump
```c
#include <stdlib.h>

int main() {
    system("tcpdump -c 1");
    return 0;
}
```

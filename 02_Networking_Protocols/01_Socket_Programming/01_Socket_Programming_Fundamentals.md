# Socket Programming Fundamentals

*Last Updated: June 21, 2025*

## Overview

This module covers the fundamental concepts of socket programming, providing the foundation for network communication in applications.

## Learning Objectives

By the end of this module, you should understand:
- Socket API overview and basic concepts
- Socket address structures and their usage
- Network byte order and endianness handling
- Socket options and flags configuration
- Proper error handling in socket programming

## Topics Covered

### Socket API Overview
- What are sockets?
- Socket types (TCP, UDP, Unix domain)
- Socket states and lifecycle
- Socket file descriptors

### Socket Address Structures
- `sockaddr` structure family
- `sockaddr_in` for IPv4
- `sockaddr_in6` for IPv6
- `sockaddr_un` for Unix domain sockets
- Address family constants (AF_INET, AF_INET6, AF_UNIX)

### Network Byte Order and Endianness
- Host byte order vs network byte order
- Conversion functions:
  - `htons()` - host to network short
  - `htonl()` - host to network long
  - `ntohs()` - network to host short
  - `ntohl()` - network to host long
- Portable programming considerations

### Socket Options and Flags
- Common socket options:
  - SO_REUSEADDR
  - SO_KEEPALIVE
  - SO_RCVBUF/SO_SNDBUF
  - SO_BROADCAST
- Setting socket options with `setsockopt()`
- Getting socket options with `getsockopt()`
- Socket flags for send/recv operations

### Error Handling
- Socket error codes and their meanings
- Platform-specific error handling
- `errno` and `WSAGetLastError()`
- Proper error checking patterns
- Graceful error recovery strategies

## Practical Exercises

1. **Basic Socket Creation**
   - Create different types of sockets
   - Explore socket properties and options

2. **Address Structure Practice**
   - Convert between string and binary IP addresses
   - Practice with different address families

3. **Endianness Conversion**
   - Implement byte order conversion examples
   - Test on different architectures

4. **Error Handling Framework**
   - Build a robust error handling system
   - Create error logging and reporting utilities

## Code Examples

### Basic Socket Creation
```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

// Create a TCP socket
int tcp_socket = socket(AF_INET, SOCK_STREAM, 0);
if (tcp_socket == -1) {
    perror("TCP socket creation failed");
    return -1;
}

// Create a UDP socket
int udp_socket = socket(AF_INET, SOCK_DGRAM, 0);
if (udp_socket == -1) {
    perror("UDP socket creation failed");
    return -1;
}
```

### Address Structure Setup
```c
struct sockaddr_in server_addr;
memset(&server_addr, 0, sizeof(server_addr));
server_addr.sin_family = AF_INET;
server_addr.sin_port = htons(8080);
server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
```

## Assessment Checklist

- [ ] Can create sockets of different types
- [ ] Understands socket address structures
- [ ] Properly handles byte order conversions
- [ ] Implements comprehensive error handling
- [ ] Configures socket options appropriately

## Next Steps

After completing this module, proceed to:
- TCP Socket Programming (Client-Side)
- TCP Socket Programming (Server-Side)

## Resources

- [Beej's Guide to Network Programming](https://beej.us/guide/bgnet/)
- "UNIX Network Programming, Volume 1" by W. Richard Stevens (Chapters 1-4)
- Man pages: socket(2), sockaddr(3), htons(3)

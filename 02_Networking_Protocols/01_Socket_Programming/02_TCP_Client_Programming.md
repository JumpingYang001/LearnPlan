# TCP Socket Programming - Client Side

*Last Updated: June 21, 2025*

## Overview

This module focuses on implementing TCP client applications using socket programming. You'll learn how to establish connections, exchange data, and handle various client-side scenarios.

## Learning Objectives

By the end of this module, you should be able to:
- Create and configure TCP client sockets
- Establish connections to TCP servers
- Send and receive data reliably
- Handle connection termination gracefully
- Implement proper error handling for client operations

## Topics Covered

### Creating TCP Sockets
- Socket creation with `socket(AF_INET, SOCK_STREAM, 0)`
- Socket configuration and options
- Client-specific socket settings

### Connecting to Servers
- Server address preparation
- Using `connect()` system call
- Connection timeout handling
- Retry mechanisms and backoff strategies

### Sending and Receiving Data
- `send()` and `recv()` functions
- Partial send/receive handling
- Buffer management
- Data serialization considerations

### Connection Termination
- Graceful shutdown with `shutdown()`
- Socket cleanup with `close()`
- Handling server-initiated disconnections
- TIME_WAIT state considerations

### Handling Connection Errors
- Connection refused scenarios
- Network unreachable errors
- Timeout handling
- Connection reset by peer

## Practical Exercises

1. **Simple TCP Client**
   - Connect to an echo server
   - Send messages and receive responses

2. **Robust Client Implementation**
   - Add retry logic and error handling
   - Implement connection timeouts

3. **File Transfer Client**
   - Send files to a server
   - Handle large file transfers

4. **Chat Client**
   - Interactive chat client
   - Handle real-time messaging

## Code Examples

### Basic TCP Client
```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>

int create_tcp_client(const char* server_ip, int port) {
    int sockfd;
    struct sockaddr_in server_addr;
    
    // Create socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        return -1;
    }
    
    // Configure server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    server_addr.sin_addr.s_addr = inet_addr(server_ip);
    
    // Connect to server
    if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connection failed");
        close(sockfd);
        return -1;
    }
    
    return sockfd;
}
```

### Reliable Send Function
```c
ssize_t send_all(int sockfd, const void* buffer, size_t length) {
    const char* ptr = (const char*)buffer;
    size_t total_sent = 0;
    
    while (total_sent < length) {
        ssize_t sent = send(sockfd, ptr + total_sent, length - total_sent, 0);
        if (sent < 0) {
            if (errno == EINTR) continue;  // Interrupted, retry
            return -1;  // Error occurred
        }
        if (sent == 0) break;  // Connection closed
        total_sent += sent;
    }
    
    return total_sent;
}
```

### Reliable Receive Function
```c
ssize_t recv_all(int sockfd, void* buffer, size_t length) {
    char* ptr = (char*)buffer;
    size_t total_received = 0;
    
    while (total_received < length) {
        ssize_t received = recv(sockfd, ptr + total_received, length - total_received, 0);
        if (received < 0) {
            if (errno == EINTR) continue;  // Interrupted, retry
            return -1;  // Error occurred
        }
        if (received == 0) break;  // Connection closed by peer
        total_received += received;
    }
    
    return total_received;
}
```

### Connection with Timeout
```c
#include <sys/select.h>
#include <fcntl.h>

int connect_with_timeout(int sockfd, const struct sockaddr* addr, socklen_t addrlen, int timeout_sec) {
    int flags, result;
    fd_set writefds;
    struct timeval timeout;
    
    // Set socket to non-blocking
    flags = fcntl(sockfd, F_GETFL, 0);
    fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);
    
    // Attempt connection
    result = connect(sockfd, addr, addrlen);
    if (result == 0) {
        // Connection succeeded immediately
        fcntl(sockfd, F_SETFL, flags);  // Restore blocking mode
        return 0;
    }
    
    if (errno != EINPROGRESS) {
        // Connection failed immediately
        return -1;
    }
    
    // Wait for connection to complete
    FD_ZERO(&writefds);
    FD_SET(sockfd, &writefds);
    timeout.tv_sec = timeout_sec;
    timeout.tv_usec = 0;
    
    result = select(sockfd + 1, NULL, &writefds, NULL, &timeout);
    if (result == 0) {
        // Timeout
        errno = ETIMEDOUT;
        return -1;
    } else if (result < 0) {
        // Select error
        return -1;
    }
    
    // Check if connection succeeded
    int error;
    socklen_t len = sizeof(error);
    if (getsockopt(sockfd, SOL_SOCKET, SO_ERROR, &error, &len) < 0) {
        return -1;
    }
    
    if (error != 0) {
        errno = error;
        return -1;
    }
    
    // Restore blocking mode
    fcntl(sockfd, F_SETFL, flags);
    return 0;
}
```

## Common Patterns

### Echo Client Pattern
```c
void echo_client_loop(int sockfd) {
    char buffer[1024];
    
    while (1) {
        printf("Enter message (or 'quit' to exit): ");
        if (!fgets(buffer, sizeof(buffer), stdin)) break;
        
        if (strncmp(buffer, "quit", 4) == 0) break;
        
        // Send message
        if (send_all(sockfd, buffer, strlen(buffer)) < 0) {
            perror("Send failed");
            break;
        }
        
        // Receive echo
        ssize_t received = recv(sockfd, buffer, sizeof(buffer) - 1, 0);
        if (received <= 0) {
            printf("Server disconnected\n");
            break;
        }
        
        buffer[received] = '\0';
        printf("Server echo: %s", buffer);
    }
}
```

## Error Handling Best Practices

1. **Always check return values** from socket functions
2. **Use errno** to get detailed error information
3. **Implement retry logic** for transient errors
4. **Clean up resources** properly on errors
5. **Provide meaningful error messages** to users

## Assessment Checklist

- [ ] Can create and configure TCP client sockets
- [ ] Successfully connects to TCP servers
- [ ] Implements reliable data transmission
- [ ] Handles connection errors gracefully
- [ ] Properly terminates connections
- [ ] Includes comprehensive error handling

## Next Steps

After mastering TCP client programming:
- Proceed to TCP Server Programming
- Explore advanced client patterns (connection pooling, reconnection strategies)
- Learn about non-blocking client implementations

## Resources

- "UNIX Network Programming, Volume 1" by W. Richard Stevens (Chapters 4-5)
- [Beej's Guide - Client-Server Background](https://beej.us/guide/bgnet/html/#client-server-background)
- Linux man pages: connect(2), send(2), recv(2)

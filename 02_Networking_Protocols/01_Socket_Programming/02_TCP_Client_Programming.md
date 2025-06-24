# TCP Socket Programming - Client Side

*Last Updated: June 21, 2025*

## Overview

This module provides comprehensive coverage of TCP client-side socket programming. TCP (Transmission Control Protocol) is a reliable, connection-oriented protocol that guarantees ordered delivery of data between applications. Understanding TCP client programming is fundamental for building networked applications that need reliable data transmission.

### What You'll Learn
- **Socket fundamentals** - How TCP sockets work under the hood
- **Connection management** - Establishing and maintaining reliable connections
- **Data transmission** - Sending and receiving data with proper error handling
- **Advanced patterns** - Real-world client implementations and best practices
- **Troubleshooting** - Common issues and their solutions

### Prerequisites
- Basic C programming knowledge
- Understanding of network concepts (IP addresses, ports)
- Familiarity with system calls and error handling
- Basic understanding of TCP/IP protocol stack

### TCP Client Communication Flow
```
Client                                    Server
  |                                         |
  |  1. socket() - Create socket           |
  |  2. connect() - Initiate connection    |
  |  ────────── SYN ──────────────────────>|
  |  <───────── SYN-ACK ──────────────────|
  |  ────────── ACK ──────────────────────>|
  |  3. Connection established             |
  |                                         |
  |  4. send() - Send data                 |
  |  ────────── Data ─────────────────────>|
  |  <───────── ACK ──────────────────────|
  |                                         |
  |  5. recv() - Receive response          |
  |  <───────── Data ─────────────────────|
  |  ────────── ACK ──────────────────────>|
  |                                         |
  |  6. close() - Close connection         |
  |  ────────── FIN ──────────────────────>|
  |  <───────── FIN-ACK ──────────────────|
  |  ────────── ACK ──────────────────────>|
  |                                         |
```

## Learning Objectives

By the end of this module, you should be able to:

### Core Skills
- **Create and configure TCP client sockets** with appropriate options and settings
- **Establish reliable connections** to TCP servers with proper error handling
- **Implement robust data transmission** that handles partial sends and receives
- **Handle connection termination** gracefully in both normal and error scenarios
- **Debug network issues** using appropriate tools and techniques

### Advanced Skills
- **Implement connection timeouts** and retry mechanisms
- **Build thread-safe client applications** for concurrent connections
- **Handle various network error conditions** with appropriate recovery strategies
- **Optimize client performance** through connection pooling and keep-alive mechanisms
- **Implement secure communications** using SSL/TLS over TCP

### Practical Applications
- Build a **file transfer client** that can handle large files reliably
- Create a **chat client** with real-time messaging capabilities
- Develop an **HTTP client** that follows redirects and handles various response codes
- Implement a **database client** with connection pooling and transaction support

## Topics Covered

### Creating TCP Sockets

#### Socket Creation Fundamentals
The `socket()` system call creates an endpoint for communication. For TCP clients, we use:
- **Domain**: `AF_INET` (IPv4) or `AF_INET6` (IPv6)
- **Type**: `SOCK_STREAM` (reliable, connection-oriented)
- **Protocol**: `0` (default TCP) or `IPPROTO_TCP`

```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

// Basic socket creation
int sockfd = socket(AF_INET, SOCK_STREAM, 0);
if (sockfd < 0) {
    perror("socket creation failed");
    exit(EXIT_FAILURE);
}
```

#### Socket Configuration and Options
TCP sockets can be configured with various options to optimize behavior:

```c
#include <sys/socket.h>

int configure_tcp_socket(int sockfd) {
    // Enable address reuse (helpful for development)
    int reuse = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        perror("setsockopt SO_REUSEADDR failed");
        return -1;
    }
    
    // Set send buffer size
    int send_buffer_size = 65536;  // 64KB
    if (setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &send_buffer_size, sizeof(send_buffer_size)) < 0) {
        perror("setsockopt SO_SNDBUF failed");
        return -1;
    }
    
    // Set receive buffer size
    int recv_buffer_size = 65536;  // 64KB
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &recv_buffer_size, sizeof(recv_buffer_size)) < 0) {
        perror("setsockopt SO_RCVBUF failed");
        return -1;
    }
    
    // Enable TCP keepalive
    int keepalive = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE, &keepalive, sizeof(keepalive)) < 0) {
        perror("setsockopt SO_KEEPALIVE failed");
        return -1;
    }
    
    // Configure keepalive parameters (Linux-specific)
    int keepidle = 60;    // Start keepalive after 60 seconds of inactivity
    int keepintvl = 10;   // Send keepalive probes every 10 seconds
    int keepcnt = 3;      // Close connection after 3 failed probes
    
    setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPIDLE, &keepidle, sizeof(keepidle));
    setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPINTVL, &keepintvl, sizeof(keepintvl));
    setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPCNT, &keepcnt, sizeof(keepcnt));
    
    // Disable Nagle's algorithm for low-latency applications
    int nodelay = 1;
    if (setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay)) < 0) {
        perror("setsockopt TCP_NODELAY failed");
        return -1;
    }
    
    return 0;
}
```

#### Client-Specific Socket Settings
```c
// Set socket timeout for send operations
struct timeval timeout;
timeout.tv_sec = 30;   // 30 seconds timeout
timeout.tv_usec = 0;

if (setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) < 0) {
    perror("setsockopt SO_SNDTIMEO failed");
}

// Set socket timeout for receive operations
if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
    perror("setsockopt SO_RCVTIMEO failed");
}

// Set socket to non-blocking mode (for advanced applications)
int flags = fcntl(sockfd, F_GETFL, 0);
if (flags >= 0) {
    fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);
}
```

### Connecting to Servers

#### Server Address Preparation
Before connecting, you must prepare the server address structure:

```c
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

// Method 1: Using IP address directly
struct sockaddr_in setup_server_addr_ip(const char* ip, int port) {
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    
    if (inet_pton(AF_INET, ip, &server_addr.sin_addr) <= 0) {
        fprintf(stderr, "Invalid IP address: %s\n", ip);
        exit(EXIT_FAILURE);
    }
    
    return server_addr;
}

// Method 2: Using hostname resolution
struct sockaddr_in setup_server_addr_hostname(const char* hostname, int port) {
    struct sockaddr_in server_addr;
    struct hostent* host_entry;
    
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    
    // Resolve hostname to IP address
    host_entry = gethostbyname(hostname);
    if (host_entry == NULL) {
        fprintf(stderr, "Failed to resolve hostname: %s\n", hostname);
        exit(EXIT_FAILURE);
    }
    
    memcpy(&server_addr.sin_addr, host_entry->h_addr_list[0], host_entry->h_length);
    return server_addr;
}

// Method 3: Modern approach using getaddrinfo (IPv4/IPv6 compatible)
int setup_server_addr_modern(const char* hostname, const char* port, struct addrinfo** result) {
    struct addrinfo hints;
    
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;      // Allow IPv4 or IPv6
    hints.ai_socktype = SOCK_STREAM;  // TCP socket
    hints.ai_flags = 0;
    hints.ai_protocol = 0;
    
    int status = getaddrinfo(hostname, port, &hints, result);
    if (status != 0) {
        fprintf(stderr, "getaddrinfo error: %s\n", gai_strerror(status));
        return -1;
    }
    
    return 0;
}
```

#### Using connect() System Call
The `connect()` system call establishes a connection to the server:

```c
#include <sys/socket.h>
#include <errno.h>

int establish_connection(int sockfd, const struct sockaddr_in* server_addr) {
    printf("Attempting to connect to %s:%d...\n", 
           inet_ntoa(server_addr->sin_addr), 
           ntohs(server_addr->sin_port));
    
    if (connect(sockfd, (struct sockaddr*)server_addr, sizeof(*server_addr)) < 0) {
        switch (errno) {
            case ECONNREFUSED:
                fprintf(stderr, "Connection refused - server not listening\n");
                break;
            case ETIMEDOUT:
                fprintf(stderr, "Connection timed out\n");
                break;
            case ENETUNREACH:
                fprintf(stderr, "Network unreachable\n");
                break;
            case EHOSTUNREACH:
                fprintf(stderr, "Host unreachable\n");
                break;
            default:
                perror("connect failed");
                break;
        }
        return -1;
    }
    
    printf("Connection established successfully!\n");
    return 0;
}
```

#### Connection Timeout Handling
Implementing timeouts prevents clients from hanging indefinitely:

```c
#include <sys/select.h>
#include <fcntl.h>

int connect_with_timeout(int sockfd, const struct sockaddr* addr, socklen_t addrlen, int timeout_seconds) {
    int flags, result, error;
    socklen_t len;
    fd_set writefds;
    struct timeval timeout;
    
    // Save original socket flags
    flags = fcntl(sockfd, F_GETFL, 0);
    if (flags < 0) {
        perror("fcntl F_GETFL failed");
        return -1;
    }
    
    // Set socket to non-blocking mode
    if (fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) < 0) {
        perror("fcntl F_SETFL failed");
        return -1;
    }
    
    // Attempt connection
    result = connect(sockfd, addr, addrlen);
    
    if (result == 0) {
        // Connection succeeded immediately (rare for remote connections)
        fcntl(sockfd, F_SETFL, flags);  // Restore original flags
        return 0;
    }
    
    if (errno != EINPROGRESS) {
        // Connection failed immediately
        fcntl(sockfd, F_SETFL, flags);
        return -1;
    }
    
    // Connection is in progress, wait for completion
    FD_ZERO(&writefds);
    FD_SET(sockfd, &writefds);
    timeout.tv_sec = timeout_seconds;
    timeout.tv_usec = 0;
    
    result = select(sockfd + 1, NULL, &writefds, NULL, &timeout);
    
    if (result == 0) {
        // Timeout occurred
        errno = ETIMEDOUT;
        fcntl(sockfd, F_SETFL, flags);
        return -1;
    }
    
    if (result < 0) {
        // select() failed
        fcntl(sockfd, F_SETFL, flags);
        return -1;
    }
    
    // Check if connection succeeded or failed
    len = sizeof(error);
    if (getsockopt(sockfd, SOL_SOCKET, SO_ERROR, &error, &len) < 0) {
        fcntl(sockfd, F_SETFL, flags);
        return -1;
    }
    
    if (error != 0) {
        // Connection failed
        errno = error;
        fcntl(sockfd, F_SETFL, flags);
        return -1;
    }
    
    // Connection succeeded
    fcntl(sockfd, F_SETFL, flags);  // Restore original flags
    return 0;
}
```

#### Retry Mechanisms and Backoff Strategies
Robust clients implement retry logic with exponential backoff:

```c
#include <unistd.h>
#include <math.h>

typedef struct {
    int max_retries;
    int base_delay_ms;
    double backoff_multiplier;
    int max_delay_ms;
    int jitter_enabled;
} retry_config_t;

int connect_with_retry(int sockfd, const struct sockaddr* addr, socklen_t addrlen, 
                      const retry_config_t* config) {
    int attempt = 0;
    int delay_ms = config->base_delay_ms;
    
    while (attempt < config->max_retries) {
        printf("Connection attempt %d/%d...\n", attempt + 1, config->max_retries);
        
        if (connect_with_timeout(sockfd, addr, addrlen, 10) == 0) {
            printf("Connection successful on attempt %d\n", attempt + 1);
            return 0;
        }
        
        attempt++;
        
        if (attempt < config->max_retries) {
            // Add jitter to prevent thundering herd
            int actual_delay = delay_ms;
            if (config->jitter_enabled) {
                int jitter = rand() % (delay_ms / 2);  // Up to 50% jitter
                actual_delay += jitter;
            }
            
            printf("Connection failed, retrying in %d ms...\n", actual_delay);
            usleep(actual_delay * 1000);  // Convert to microseconds
            
            // Exponential backoff
            delay_ms = (int)(delay_ms * config->backoff_multiplier);
            if (delay_ms > config->max_delay_ms) {
                delay_ms = config->max_delay_ms;
            }
        }
    }
    
    fprintf(stderr, "All connection attempts failed after %d tries\n", config->max_retries);
    return -1;
}

// Usage example
void example_retry_connection() {
    retry_config_t config = {
        .max_retries = 5,
        .base_delay_ms = 1000,      // Start with 1 second
        .backoff_multiplier = 2.0,   // Double each time
        .max_delay_ms = 30000,       // Cap at 30 seconds
        .jitter_enabled = 1
    };
    
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in server_addr = setup_server_addr_ip("192.168.1.100", 8080);
    
    if (connect_with_retry(sockfd, (struct sockaddr*)&server_addr, 
                          sizeof(server_addr), &config) == 0) {
        printf("Successfully connected to server!\n");
        // Use the connection...
        close(sockfd);
    } else {
        printf("Failed to establish connection\n");
        close(sockfd);
    }
}
```

### Sending and Receiving Data

#### Understanding send() and recv() Functions
TCP provides reliable data transmission, but the `send()` and `recv()` functions don't guarantee that all data is sent or received in a single call.

```c
#include <sys/socket.h>
#include <errno.h>

// send() function signature
ssize_t send(int sockfd, const void *buf, size_t len, int flags);

// recv() function signature  
ssize_t recv(int sockfd, void *buf, size_t len, int flags);
```

**Important Characteristics:**
- **Partial Operations**: May send/receive fewer bytes than requested
- **Blocking Behavior**: By default, calls block until some data is transferred
- **Return Values**: Number of bytes transferred, 0 for connection closed, -1 for error
- **Interruption**: Can be interrupted by signals (EINTR)

#### Robust Send Implementation
```c
#include <sys/socket.h>
#include <errno.h>

ssize_t send_all(int sockfd, const void* buffer, size_t length) {
    const char* ptr = (const char*)buffer;
    size_t total_sent = 0;
    
    while (total_sent < length) {
        ssize_t sent = send(sockfd, ptr + total_sent, length - total_sent, 0);
        
        if (sent < 0) {
            if (errno == EINTR) {
                continue;  // Interrupted by signal, retry
            } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Would block in non-blocking mode, wait and retry
                usleep(1000);  // Wait 1ms
                continue;
            } else {
                perror("send failed");
                return -1;  // Actual error occurred
            }
        }
        
        if (sent == 0) {
            // This shouldn't happen with send(), but handle it
            fprintf(stderr, "send() returned 0 - connection may be closed\n");
            break;
        }
        
        total_sent += sent;
    }
    
    return total_sent;
}

// Enhanced version with timeout and progress callback
typedef void (*progress_callback_t)(size_t sent, size_t total);

ssize_t send_all_enhanced(int sockfd, const void* buffer, size_t length, 
                         int timeout_seconds, progress_callback_t progress_cb) {
    const char* ptr = (const char*)buffer;
    size_t total_sent = 0;
    time_t start_time = time(NULL);
    
    while (total_sent < length) {
        // Check timeout
        if (timeout_seconds > 0 && (time(NULL) - start_time) > timeout_seconds) {
            errno = ETIMEDOUT;
            return -1;
        }
        
        ssize_t sent = send(sockfd, ptr + total_sent, length - total_sent, 0);
        
        if (sent < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                usleep(1000);
                continue;
            }
            return -1;
        }
        
        if (sent == 0) break;
        
        total_sent += sent;
        
        // Progress callback
        if (progress_cb) {
            progress_cb(total_sent, length);
        }
    }
    
    return total_sent;
}

// Progress callback example
void send_progress_callback(size_t sent, size_t total) {
    double percentage = (double)sent / total * 100.0;
    printf("\rSending progress: %.1f%% (%zu/%zu bytes)", percentage, sent, total);
    fflush(stdout);
}
```

#### Robust Receive Implementation
```c
ssize_t recv_all(int sockfd, void* buffer, size_t length) {
    char* ptr = (char*)buffer;
    size_t total_received = 0;
    
    while (total_received < length) {
        ssize_t received = recv(sockfd, ptr + total_received, 
                               length - total_received, 0);
        
        if (received < 0) {
            if (errno == EINTR) {
                continue;  // Interrupted by signal, retry
            } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Would block in non-blocking mode
                usleep(1000);
                continue;
            } else {
                perror("recv failed");
                return -1;  // Actual error occurred
            }
        }
        
        if (received == 0) {
            // Connection closed by peer
            printf("Connection closed by peer (received %zu of %zu bytes)\n", 
                   total_received, length);
            breaks;
        }
        
        total_received += received;
    }
    
    return total_received;
}

// Receive with timeout using select()
ssize_t recv_with_timeout(int sockfd, void* buffer, size_t length, int timeout_seconds) {
    fd_set readfds;
    struct timeval timeout;
    
    FD_ZERO(&readfds);
    FD_SET(sockfd, &readfds);
    timeout.tv_sec = timeout_seconds;
    timeout.tv_usec = 0;
    
    int result = select(sockfd + 1, &readfds, NULL, NULL, &timeout);
    
    if (result == 0) {
        errno = ETIMEDOUT;
        return -1;  // Timeout
    }
    
    if (result < 0) {
        return -1;  // select() error
    }
    
    // Data is available, now receive it
    return recv(sockfd, buffer, length, 0);
}

// Receive line-by-line (useful for text protocols)
ssize_t recv_line(int sockfd, char* buffer, size_t buffer_size) {
    size_t total_received = 0;
    char ch;
    
    while (total_received < buffer_size - 1) {
        ssize_t received = recv(sockfd, &ch, 1, 0);
        
        if (received < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        
        if (received == 0) {
            break;  // Connection closed
        }
        
        buffer[total_received++] = ch;
        
        if (ch == '\n') {
            break;  // End of line
        }
    }
    
    buffer[total_received] = '\0';
    return total_received;
}
```

#### Buffer Management Strategies
```c
#include <stdlib.h>

typedef struct {
    char* data;
    size_t size;
    size_t capacity;
} dynamic_buffer_t;

dynamic_buffer_t* buffer_create(size_t initial_capacity) {
    dynamic_buffer_t* buffer = malloc(sizeof(dynamic_buffer_t));
    if (!buffer) return NULL;
    
    buffer->data = malloc(initial_capacity);
    if (!buffer->data) {
        free(buffer);
        return NULL;
    }
    
    buffer->size = 0;
    buffer->capacity = initial_capacity;
    return buffer;
}

int buffer_append(dynamic_buffer_t* buffer, const void* data, size_t length) {
    if (buffer->size + length > buffer->capacity) {
        // Resize buffer
        size_t new_capacity = buffer->capacity * 2;
        if (new_capacity < buffer->size + length) {
            new_capacity = buffer->size + length;
        }
        
        char* new_data = realloc(buffer->data, new_capacity);
        if (!new_data) {
            return -1;  // Allocation failed
        }
        
        buffer->data = new_data;
        buffer->capacity = new_capacity;
    }
    
    memcpy(buffer->data + buffer->size, data, length);
    buffer->size += length;
    return 0;
}

void buffer_destroy(dynamic_buffer_t* buffer) {
    if (buffer) {
        free(buffer->data);
        free(buffer);
    }
}

// Circular buffer for streaming data
typedef struct {
    char* data;
    size_t capacity;
    size_t head;
    size_t tail;
    size_t count;
} circular_buffer_t;

circular_buffer_t* circular_buffer_create(size_t capacity) {
    circular_buffer_t* buffer = malloc(sizeof(circular_buffer_t));
    if (!buffer) return NULL;
    
    buffer->data = malloc(capacity);
    if (!buffer->data) {
        free(buffer);
        return NULL;
    }
    
    buffer->capacity = capacity;
    buffer->head = 0;
    buffer->tail = 0;
    buffer->count = 0;
    return buffer;
}
```

#### Data Serialization Considerations
```c
#include <arpa/inet.h>  // For htonl, ntohl, htons, ntohs

// Simple message protocol
typedef struct {
    uint32_t message_type;
    uint32_t message_length;
    char data[];  // Variable length data
} message_header_t;

// Send structured message
int send_message(int sockfd, uint32_t type, const void* data, uint32_t length) {
    message_header_t header;
    
    // Convert to network byte order
    header.message_type = htonl(type);
    header.message_length = htonl(length);
    
    // Send header
    if (send_all(sockfd, &header, sizeof(header)) != sizeof(header)) {
        return -1;
    }
    
    // Send data if present
    if (length > 0 && data) {
        if (send_all(sockfd, data, length) != length) {
            return -1;
        }
    }
    
    return 0;
}

// Receive structured message
int recv_message(int sockfd, uint32_t* type, void** data, uint32_t* length) {
    message_header_t header;
    
    // Receive header
    if (recv_all(sockfd, &header, sizeof(header)) != sizeof(header)) {
        return -1;
    }
    
    // Convert from network byte order
    *type = ntohl(header.message_type);
    *length = ntohl(header.message_length);
    
    // Validate message length
    if (*length > MAX_MESSAGE_SIZE) {
        fprintf(stderr, "Message too large: %u bytes\n", *length);
        return -1;
    }
    
    if (*length > 0) {
        *data = malloc(*length);
        if (!*data) {
            return -1;
        }
        
        if (recv_all(sockfd, *data, *length) != *length) {
            free(*data);
            *data = NULL;
            return -1;
        }
    } else {
        *data = NULL;
    }
    
    return 0;
}

// JSON serialization example (requires cJSON library)
#ifdef USE_JSON
#include <cjson/cJSON.h>

int send_json_message(int sockfd, const cJSON* json) {
    char* json_string = cJSON_Print(json);
    if (!json_string) {
        return -1;
    }
    
    uint32_t length = strlen(json_string);
    int result = send_message(sockfd, MSG_TYPE_JSON, json_string, length);
    
    free(json_string);
    return result;
}

cJSON* recv_json_message(int sockfd) {
    uint32_t type, length;
    void* data;
    
    if (recv_message(sockfd, &type, &data, &length) < 0) {
        return NULL;
    }
    
    if (type != MSG_TYPE_JSON) {
        free(data);
        return NULL;
    }
    
    cJSON* json = cJSON_ParseWithLength((char*)data, length);
    free(data);
    
    return json;
}
#endif
```

### Connection Termination

#### Graceful Shutdown with shutdown()
The `shutdown()` system call provides fine-grained control over connection termination:

```c
#include <sys/socket.h>

// shutdown() parameters:
// SHUT_RD (0)   - Disable further receive operations
// SHUT_WR (1)   - Disable further send operations  
// SHUT_RDWR (2) - Disable both send and receive operations

int graceful_shutdown(int sockfd) {
    printf("Initiating graceful shutdown...\n");
    
    // Step 1: Stop sending data
    if (shutdown(sockfd, SHUT_WR) < 0) {
        perror("shutdown SHUT_WR failed");
        return -1;
    }
    
    // Step 2: Read any remaining data from the peer
    char buffer[1024];
    ssize_t bytes_received;
    
    printf("Draining remaining data from peer...\n");
    while ((bytes_received = recv(sockfd, buffer, sizeof(buffer), 0)) > 0) {
        printf("Received %zd bytes during shutdown\n", bytes_received);
        // Process or discard remaining data
    }
    
    if (bytes_received < 0) {
        perror("recv during shutdown failed");
        return -1;
    }
    
    printf("Peer has closed its side of the connection\n");
    return 0;
}

// Complete shutdown sequence
int complete_shutdown(int sockfd) {
    // Graceful shutdown
    if (graceful_shutdown(sockfd) < 0) {
        fprintf(stderr, "Graceful shutdown failed, forcing close\n");
    }
    
    // Close the socket
    if (close(sockfd) < 0) {
        perror("close failed");
        return -1;
    }
    
    printf("Connection closed successfully\n");
    return 0;
}
```

#### Socket Cleanup with close()
```c
#include <unistd.h>

// Simple close
int simple_close(int sockfd) {
    if (close(sockfd) < 0) {
        perror("close failed");
        return -1;
    }
    return 0;
}

// Close with error handling and logging
int robust_close(int sockfd) {
    printf("Closing socket %d...\n", sockfd);
    
    // Get socket statistics before closing
    struct tcp_info tcp_info;
    socklen_t tcp_info_len = sizeof(tcp_info);
    
    if (getsockopt(sockfd, IPPROTO_TCP, TCP_INFO, &tcp_info, &tcp_info_len) == 0) {
        printf("Connection statistics:\n");
        printf("  State: %u\n", tcp_info.tcpi_state);
        printf("  Retransmits: %u\n", tcp_info.tcpi_retransmits);
        printf("  RTT: %u microseconds\n", tcp_info.tcpi_rtt);
    }
    
    if (close(sockfd) < 0) {
        perror("close failed");
        return -1;
    }
    
    printf("Socket closed successfully\n");
    return 0;
}
```

#### Handling Server-Initiated Disconnections
```c
#include <signal.h>

// Handle broken pipe signal (SIGPIPE)
void sigpipe_handler(int sig) {
    printf("Received SIGPIPE - peer closed connection\n");
    // Don't exit, just log the event
}

// Detect server disconnection during recv()
int detect_server_disconnect(int sockfd) {
    char buffer[1024];
    
    // Install SIGPIPE handler
    signal(SIGPIPE, sigpipe_handler);
    
    while (1) {
        ssize_t received = recv(sockfd, buffer, sizeof(buffer), 0);
        
        if (received > 0) {
            // Normal data received
            printf("Received %zd bytes\n", received);
            // Process the data...
            
        } else if (received == 0) {
            // Server closed the connection gracefully
            printf("Server closed connection gracefully\n");
            break;
            
        } else {
            // Error occurred
            if (errno == ECONNRESET) {
                printf("Connection reset by peer\n");
            } else if (errno == EPIPE) {
                printf("Broken pipe - peer closed connection\n");
            } else {
                perror("recv failed");
            }
            break;
        }
    }
    
    return 0;
}

// Monitor connection health with keepalive
int monitor_connection_health(int sockfd) {
    int keepalive = 1;
    int keepidle = 30;    // Start probing after 30 seconds
    int keepintvl = 5;    // Probe every 5 seconds
    int keepcnt = 3;      // Give up after 3 failed probes
    
    // Enable keepalive
    if (setsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE, &keepalive, sizeof(keepalive)) < 0) {
        perror("setsockopt SO_KEEPALIVE failed");
        return -1;
    }
    
    // Configure keepalive parameters
    setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPIDLE, &keepidle, sizeof(keepidle));
    setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPINTVL, &keepintvl, sizeof(keepintvl));
    setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPCNT, &keepcnt, sizeof(keepcnt));
    
    printf("Keepalive enabled: idle=%ds, interval=%ds, count=%d\n", 
           keepidle, keepintvl, keepcnt);
    
    return 0;
}
```

#### TIME_WAIT State Considerations
```c
// Understanding TIME_WAIT state
void explain_time_wait() {
    printf("TIME_WAIT State Information:\n");
    printf("- Occurs when the local side initiates connection closure\n");
    printf("- Lasts for 2 * MSL (Maximum Segment Lifetime)\n");
    printf("- On Linux, typically 60 seconds\n");
    printf("- Prevents port reuse during this period\n");
    printf("- Use SO_REUSEADDR to allow immediate port reuse\n");
}

// Configure socket to handle TIME_WAIT
int configure_for_time_wait(int sockfd) {
    int reuse = 1;
    
    // Allow address reuse
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        perror("setsockopt SO_REUSEADDR failed");
        return -1;
    }
    
    // Allow port reuse (Linux specific)
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(reuse)) < 0) {
        // Not all systems support SO_REUSEPORT
        if (errno != ENOPROTOOPT) {
            perror("setsockopt SO_REUSEPORT failed");
        }
    }
    
    return 0;
}

// Linger option for controlling close behavior
int configure_linger(int sockfd, int enable, int timeout_seconds) {
    struct linger linger_opt;
    
    linger_opt.l_onoff = enable;
    linger_opt.l_linger = timeout_seconds;
    
    if (setsockopt(sockfd, SOL_SOCKET, SO_LINGER, &linger_opt, sizeof(linger_opt)) < 0) {
        perror("setsockopt SO_LINGER failed");
        return -1;
    }
    
    if (enable) {
        printf("Linger enabled: will wait up to %d seconds for pending data\n", timeout_seconds);
    } else {
        printf("Linger disabled: close() returns immediately\n");
    }
    
    return 0;
}
```

### Handling Connection Errors

#### Connection Refused Scenarios
```c
#include <errno.h>

void handle_connection_refused(const char* server_ip, int port) {
    printf("Connection refused to %s:%d\n", server_ip, port);
    printf("Possible causes:\n");
    printf("1. Server is not running\n");
    printf("2. Server is not listening on port %d\n", port);
    printf("3. Firewall is blocking the connection\n");
    printf("4. Port is already in use by another service\n");
    
    printf("Troubleshooting steps:\n");
    printf("1. Verify server is running: ps aux | grep server_name\n");
    printf("2. Check if port is listening: netstat -tlnp | grep %d\n", port);
    printf("3. Test with telnet: telnet %s %d\n", server_ip, port);
    printf("4. Check firewall rules: iptables -L\n");
}
```

#### Network Unreachable Errors
```c
void handle_network_unreachable(const char* server_ip) {
    printf("Network unreachable to %s\n", server_ip);
    printf("Possible causes:\n");
    printf("1. No route to destination network\n");
    printf("2. Network interface is down\n");
    printf("3. Routing table misconfiguration\n");
    printf("4. Network cable unplugged\n");
    
    printf("Troubleshooting steps:\n");
    printf("1. Check network interface: ip link show\n");
    printf("2. Check routing table: ip route show\n");
    printf("3. Test connectivity: ping %s\n", server_ip);
    printf("4. Check DNS resolution: nslookup %s\n", server_ip);
}
```

#### Comprehensive Error Handling
```c
typedef enum {
    CONN_SUCCESS = 0,
    CONN_REFUSED,
    CONN_TIMEOUT,
    CONN_UNREACHABLE,
    CONN_RESET,
    CONN_OTHER_ERROR
} connection_result_t;

connection_result_t connect_with_error_handling(int sockfd, const struct sockaddr* addr, socklen_t addrlen) {
    if (connect(sockfd, addr, addrlen) == 0) {
        return CONN_SUCCESS;
    }
    
    switch (errno) {
        case ECONNREFUSED:
            fprintf(stderr, "Connection refused\n");
            return CONN_REFUSED;
            
        case ETIMEDOUT:
            fprintf(stderr, "Connection timed out\n");
            return CONN_TIMEOUT;
            
        case ENETUNREACH:
            fprintf(stderr, "Network unreachable\n");
            return CONN_UNREACHABLE;
            
        case EHOSTUNREACH:
            fprintf(stderr, "Host unreachable\n");
            return CONN_UNREACHABLE;
            
        case ECONNRESET:
            fprintf(stderr, "Connection reset by peer\n");
            return CONN_RESET;
            
        case EADDRINUSE:
            fprintf(stderr, "Address already in use\n");
            return CONN_OTHER_ERROR;
            
        case EACCES:
            fprintf(stderr, "Permission denied\n");
            return CONN_OTHER_ERROR;
            
        default:
            fprintf(stderr, "Connection failed: %s\n", strerror(errno));
            return CONN_OTHER_ERROR;
    }
}

// Connection with retry and different error handling
int robust_connect(const char* hostname, int port, int max_retries) {
    struct addrinfo hints, *result, *rp;
    int sockfd = -1;
    char port_str[16];
    
    snprintf(port_str, sizeof(port_str), "%d", port);
    
    // Setup hints for getaddrinfo
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    
    // Resolve hostname
    int status = getaddrinfo(hostname, port_str, &hints, &result);
    if (status != 0) {
        fprintf(stderr, "getaddrinfo failed: %s\n", gai_strerror(status));
        return -1;
    }
    
    // Try each address until one succeeds
    for (rp = result; rp != NULL; rp = rp->ai_next) {
        sockfd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sockfd == -1) continue;
        
        // Configure socket
        configure_for_time_wait(sockfd);
        
        // Try to connect with retries
        for (int attempt = 0; attempt < max_retries; attempt++) {
            connection_result_t conn_result = connect_with_error_handling(sockfd, rp->ai_addr, rp->ai_addrlen);
            
            if (conn_result == CONN_SUCCESS) {
                printf("Connected successfully on attempt %d\n", attempt + 1);
                freeaddrinfo(result);
                return sockfd;
            }
            
            if (conn_result == CONN_REFUSED || conn_result == CONN_UNREACHABLE) {
                // These errors are unlikely to be transient
                if (attempt == 0) {
                    printf("Permanent error detected, skipping retries\n");
                    break;
                }
            }
            
            if (attempt < max_retries - 1) {
                printf("Retrying in %d seconds...\n", attempt + 1);
                sleep(attempt + 1);  // Exponential backoff
            }
        }
        
        close(sockfd);
        sockfd = -1;
    }
    
    freeaddrinfo(result);
    
    if (sockfd == -1) {
        fprintf(stderr, "Could not connect to %s:%d after %d attempts\n", 
                hostname, port, max_retries);
    }
    
    return sockfd;
}
```

## Practical Exercises

### Exercise 1: Simple TCP Echo Client
Build a basic client that connects to an echo server and exchanges messages.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <server_ip> <port>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    
    const char* server_ip = argv[1];
    int port = atoi(argv[2]);
    
    // Create socket
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
    
    // Setup server address
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    server_addr.sin_addr.s_addr = inet_addr(server_ip);
    
    // Connect to server
    if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    
    printf("Connected to echo server at %s:%d\n", server_ip, port);
    
    // Interactive loop
    char buffer[1024];
    while (1) {
        printf("Enter message (or 'quit' to exit): ");
        if (!fgets(buffer, sizeof(buffer), stdin)) {
            break;
        }
        
        // Remove newline
        buffer[strcspn(buffer, "\n")] = 0;
        
        if (strcmp(buffer, "quit") == 0) {
            break;
        }
        
        // Send message
        if (send(sockfd, buffer, strlen(buffer), 0) < 0) {
            perror("send failed");
            break;
        }
        
        // Receive echo
        ssize_t received = recv(sockfd, buffer, sizeof(buffer) - 1, 0);
        if (received <= 0) {
            printf("Server disconnected\n");
            break;
        }
        
        buffer[received] = '\0';
        printf("Server echo: %s\n", buffer);
    }
    
    close(sockfd);
    printf("Connection closed\n");
    return 0;
}
```

**Challenge Extensions:**
- Add connection timeout handling
- Implement retry logic on connection failure
- Add message timestamps
- Handle partial send/receive operations

### Exercise 2: Robust HTTP Client
Create an HTTP client that can download web pages with proper error handling.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

typedef struct {
    int status_code;
    char* headers;
    char* body;
    size_t body_length;
} http_response_t;

int parse_url(const char* url, char* hostname, char* path, int* port) {
    // Simple URL parser for http://hostname:port/path
    if (strncmp(url, "http://", 7) != 0) {
        return -1;
    }
    
    const char* start = url + 7;
    const char* path_start = strchr(start, '/');
    const char* port_start = strchr(start, ':');
    
    // Extract hostname
    int hostname_len;
    if (port_start && (!path_start || port_start < path_start)) {
        hostname_len = port_start - start;
        *port = atoi(port_start + 1);
    } else {
        hostname_len = path_start ? path_start - start : strlen(start);
        *port = 80;  // Default HTTP port
    }
    
    strncpy(hostname, start, hostname_len);
    hostname[hostname_len] = '\0';
    
    // Extract path
    if (path_start) {
        strcpy(path, path_start);
    } else {
        strcpy(path, "/");
    }
    
    return 0;
}

int http_get(const char* hostname, int port, const char* path, http_response_t* response) {
    // Create socket
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        return -1;
    }
    
    // Resolve hostname
    struct hostent* host = gethostbyname(hostname);
    if (!host) {
        close(sockfd);
        return -1;
    }
    
    // Setup server address
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    memcpy(&server_addr.sin_addr, host->h_addr_list[0], host->h_length);
    
    // Connect with timeout
    if (connect_with_timeout(sockfd, (struct sockaddr*)&server_addr, 
                            sizeof(server_addr), 10) < 0) {
        close(sockfd);
        return -1;
    }
    
    // Build HTTP request
    char request[2048];
    snprintf(request, sizeof(request),
             "GET %s HTTP/1.1\r\n"
             "Host: %s\r\n"
             "User-Agent: SimpleHTTPClient/1.0\r\n"
             "Connection: close\r\n"
             "\r\n", path, hostname);
    
    // Send request
    if (send_all(sockfd, request, strlen(request)) < 0) {
        close(sockfd);
        return -1;
    }
    
    // Receive response
    dynamic_buffer_t* buffer = buffer_create(4096);
    if (!buffer) {
        close(sockfd);
        return -1;
    }
    
    char chunk[1024];
    ssize_t received;
    while ((received = recv(sockfd, chunk, sizeof(chunk), 0)) > 0) {
        if (buffer_append(buffer, chunk, received) < 0) {
            buffer_destroy(buffer);
            close(sockfd);
            return -1;
        }
    }
    
    close(sockfd);
    
    // Parse response
    char* response_text = buffer->data;
    char* header_end = strstr(response_text, "\r\n\r\n");
    if (!header_end) {
        buffer_destroy(buffer);
        return -1;
    }
    
    // Extract status code
    if (sscanf(response_text, "HTTP/1.%*d %d", &response->status_code) != 1) {
        buffer_destroy(buffer);
        return -1;
    }
    
    // Split headers and body
    *header_end = '\0';
    response->headers = strdup(response_text);
    response->body = header_end + 4;  // Skip \r\n\r\n
    response->body_length = buffer->size - (response->body - response_text);
    
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <url>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    
    char hostname[256], path[1024];
    int port;
    
    if (parse_url(argv[1], hostname, path, &port) < 0) {
        fprintf(stderr, "Invalid URL format\n");
        exit(EXIT_FAILURE);   
    }
    
    printf("Connecting to %s:%d%s\n", hostname, port, path);
    
    http_response_t response;
    if (http_get(hostname, port, path, &response) < 0) {
        fprintf(stderr, "HTTP request failed\n");
        exit(EXIT_FAILURE);
    }
    
    printf("HTTP Status: %d\n", response.status_code);
    printf("Headers:\n%s\n", response.headers);
    printf("Body (%zu bytes):\n%.*s\n", 
           response.body_length, (int)response.body_length, response.body);
    
    free(response.headers);
    return 0;
}
```

### Exercise 3: File Transfer Client
Implement a client that can upload files to a server with progress tracking.

```c
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

typedef struct {
    uint32_t file_size;
    char filename[256];
} file_header_t;

void upload_progress(size_t sent, size_t total) {
    static time_t last_update = 0;
    time_t now = time(NULL);
    
    // Update progress every second
    if (now > last_update) {
        double percentage = (double)sent / total * 100.0;
        printf("\rUpload progress: %.1f%% (%zu/%zu bytes)", 
               percentage, sent, total);
        fflush(stdout);
        last_update = now;
    }
}

int upload_file(int sockfd, const char* filepath) {
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        perror("fopen failed");
        return -1;
    }
    
    // Get file size
    struct stat st;
    if (stat(filepath, &st) < 0) {
        perror("stat failed");
        fclose(file);
        return -1;
    }
    
    // Send file header
    file_header_t header;
    header.file_size = htonl(st.st_size);
    strncpy(header.filename, basename(filepath), sizeof(header.filename) - 1);
    header.filename[sizeof(header.filename) - 1] = '\0';
    
    if (send_all(sockfd, &header, sizeof(header)) != sizeof(header)) {
        fprintf(stderr, "Failed to send file header\n");
        fclose(file);
        return -1;
    }
    
    // Send file content
    char buffer[8192];
    size_t total_sent = 0;
    size_t bytes_read;
    
    while ((bytes_read = fread(buffer, 1, sizeof(buffer), file)) > 0) {
        if (send_all_enhanced(sockfd, buffer, bytes_read, 30, upload_progress) != bytes_read) {
            fprintf(stderr, "\nFailed to send file data\n");
            fclose(file);
            return -1;
        }
        total_sent += bytes_read;
    }
    
    fclose(file);
    printf("\nFile upload completed: %zu bytes\n", total_sent);
    return 0;
}
```

### Exercise 4: Multi-threaded Chat Client
Build a chat client that can send and receive messages simultaneously.

```c
#include <pthread.h>

typedef struct {
    int sockfd;
    char username[32];
} chat_context_t;

void* receive_thread(void* arg) {
    chat_context_t* ctx = (chat_context_t*)arg;
    char message[512];
    
    while (1) {
        ssize_t received = recv_line(ctx->sockfd, message, sizeof(message));
        if (received <= 0) {
            printf("\nDisconnected from server\n");
            break;
        }
        
        printf("\r%s\n", message);
        printf("[%s]: ", ctx->username);
        fflush(stdout);
    }
    
    return NULL;
}

int chat_client(const char* server_ip, int port, const char* username) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("socket creation failed");
        return -1;
    }
    
    struct sockaddr_in server_addr = setup_server_addr_ip(server_ip, port);
    
    if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect failed");
        close(sockfd);
        return -1;
    }
    
    // Send username to server
    if (send_all(sockfd, username, strlen(username)) < 0) {
        perror("Failed to send username");
        close(sockfd);
        return -1;
    }
    
    // Setup chat context
    chat_context_t ctx = { .sockfd = sockfd };
    strncpy(ctx.username, username, sizeof(ctx.username) - 1);
    
    // Start receive thread
    pthread_t receive_tid;
    if (pthread_create(&receive_tid, NULL, receive_thread, &ctx) != 0) {
        perror("pthread_create failed");
        close(sockfd);
        return -1;
    }
    
    // Main input loop
    char message[512];
    printf("Connected to chat server. Type messages (or 'quit' to exit):\n");
    
    while (1) {
        printf("[%s]: ", username);
        if (!fgets(message, sizeof(message), stdin)) {
            break;
        }
        
        // Remove newline
        message[strcspn(message, "\n")] = 0;
        
        if (strcmp(message, "quit") == 0) {
            break;
        }
        
        if (send_all(sockfd, message, strlen(message)) < 0) {
            perror("send failed");
            break;
        }
    }
    
    // Cleanup
    pthread_cancel(receive_tid);
    pthread_join(receive_tid, NULL);
    close(sockfd);
    
    return 0;
}
```

### Exercise 5: Connection Pool Implementation
Advanced exercise: Implement a connection pool for database-like connections.

```c
typedef struct connection {
    int sockfd;
    int in_use;
    time_t last_used;
    struct connection* next;
} connection_t;

typedef struct {
    connection_t* connections;
    int pool_size;
    int active_connections;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    char server_ip[16];
    int server_port;
} connection_pool_t;

connection_pool_t* pool_create(const char* server_ip, int port, int size) {
    connection_pool_t* pool = malloc(sizeof(connection_pool_t));
    if (!pool) return NULL;
    
    pool->connections = NULL;
    pool->pool_size = size;
    pool->active_connections = 0;
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->condition, NULL);
    strcpy(pool->server_ip, server_ip);
    pool->server_port = port;
    
    return pool;
}

connection_t* pool_get_connection(connection_pool_t* pool, int timeout_seconds) {
    pthread_mutex_lock(&pool->mutex);
    
    time_t start_time = time(NULL);
    
    while (1) {
        // Look for available connection
        connection_t* conn = pool->connections;
        while (conn) {
            if (!conn->in_use) {
                conn->in_use = 1;
                conn->last_used = time(NULL);
                pthread_mutex_unlock(&pool->mutex);
                return conn;
            }
            conn = conn->next;
        }
        
        // Create new connection if pool not full
        if (pool->active_connections < pool->pool_size) {
            connection_t* new_conn = malloc(sizeof(connection_t));
            if (new_conn) {
                new_conn->sockfd = create_connection(pool->server_ip, pool->server_port);
                if (new_conn->sockfd >= 0) {
                    new_conn->in_use = 1;
                    new_conn->last_used = time(NULL);
                    new_conn->next = pool->connections;
                    pool->connections = new_conn;
                    pool->active_connections++;
                    pthread_mutex_unlock(&pool->mutex);
                    return new_conn;
                }
                free(new_conn);
            }
        }
        
        // Check timeout
        if (time(NULL) - start_time >= timeout_seconds) {
            pthread_mutex_unlock(&pool->mutex);
            return NULL;
        }
        
        // Wait for connection to become available
        struct timespec wait_time;
        wait_time.tv_sec = time(NULL) + 1;
        wait_time.tv_nsec = 0;
        pthread_cond_timedwait(&pool->condition, &pool->mutex, &wait_time);
    }
}

void pool_release_connection(connection_pool_t* pool, connection_t* conn) {
    pthread_mutex_lock(&pool->mutex);
    conn->in_use = 0;
    conn->last_used = time(NULL);
    pthread_cond_signal(&pool->condition);
    pthread_mutex_unlock(&pool->mutex);
}
```

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

### Basic Skills ✓
- [ ] **Socket Creation**: Can create TCP sockets with proper error handling
- [ ] **Connection Establishment**: Successfully connects to servers using different addressing methods
- [ ] **Data Transmission**: Implements reliable send/receive with partial operation handling
- [ ] **Error Handling**: Properly handles and recovers from common network errors
- [ ] **Resource Management**: Correctly closes sockets and cleans up resources

### Intermediate Skills ✓
- [ ] **Timeout Handling**: Implements connection and operation timeouts
- [ ] **Retry Logic**: Uses exponential backoff and jitter for connection retries
- [ ] **Protocol Implementation**: Can implement simple text-based protocols
- [ ] **Buffer Management**: Efficiently handles data buffering and serialization
- [ ] **Connection Health**: Monitors connection state and handles disconnections

### Advanced Skills ✓
- [ ] **Multi-threading**: Builds thread-safe clients with concurrent operations
- [ ] **Connection Pooling**: Implements connection reuse for performance
- [ ] **SSL/TLS Integration**: Adds secure communications over TCP
- [ ] **Performance Optimization**: Optimizes throughput and latency
- [ ] **Production Readiness**: Includes logging, monitoring, and graceful shutdown

### Practical Applications ✓
- [ ] **Echo Client**: Simple request-response client
- [ ] **HTTP Client**: Web client with redirect handling
- [ ] **File Transfer**: Large file upload/download with progress
- [ ] **Chat Client**: Real-time messaging application
- [ ] **Database Client**: Connection pooling and transaction support

### Debugging and Troubleshooting ✓
- [ ] **Network Tools**: Uses tcpdump, netstat, ss for network analysis
- [ ] **Socket Options**: Configures sockets for different use cases  
- [ ] **Error Analysis**: Diagnoses and fixes common networking issues
- [ ] **Performance Profiling**: Identifies and resolves bottlenecks
- [ ] **Security Analysis**: Understands and mitigates security risks

## Common Patterns and Best Practices

### Connection Management Patterns
```c
// Pattern 1: Simple Connect-Use-Close
int simple_client_pattern(const char* server, int port) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    // Configure socket...
    // Connect...
    // Use connection...
    close(sockfd);
    return 0;
}

// Pattern 2: Persistent Connection
typedef struct {
    int sockfd;
    char server[256];
    int port;
    int connected;
} persistent_client_t;

int persistent_client_ensure_connected(persistent_client_t* client) {
    if (client->connected) {
        // Test connection health
        if (test_connection_health(client->sockfd)) {
            return 0;  // Still connected
        }
        close(client->sockfd);
        client->connected = 0;
    }
    
    // Reconnect
    client->sockfd = robust_connect(client->server, client->port, 3);
    if (client->sockfd >= 0) {
        client->connected = 1;
        return 0;
    }
    
    return -1;
}

// Pattern 3: Connection Pool
connection_t* get_pooled_connection(connection_pool_t* pool) {
    connection_t* conn = pool_get_connection(pool, 5);  // 5 second timeout
    if (!conn) {
        return NULL;
    }
    
    // Test connection before use
    if (!test_connection_health(conn->sockfd)) {
        pool_remove_connection(pool, conn);
        return get_pooled_connection(pool);  // Retry
    }
    
    return conn;
}
```

### Error Handling Patterns
```c
// Pattern 1: Retry with Backoff
int retry_operation(int (*operation)(void*), void* arg, int max_retries) {
    for (int attempt = 0; attempt < max_retries; attempt++) {
        int result = operation(arg);
        if (result == 0) {
            return 0;  // Success
        }
        
        if (is_permanent_error(errno)) {
             return -1;  // Don't retry permanent errors
        }
        
        if (attempt < max_retries - 1) {
            int delay_ms = (1 << attempt) * 1000;  // Exponential backoff
            usleep(delay_ms * 1000);
        }
    }
    
    return -1;  // All retries failed
}

// Pattern 2: Circuit Breaker
typedef struct {
    enum { CLOSED, OPEN, HALF_OPEN } state;
    int failure_count;
    int failure_threshold;
    time_t last_failure_time;
    int timeout_seconds;
} circuit_breaker_t;

int circuit_breaker_call(circuit_breaker_t* cb, int (*operation)(void*), void* arg) {
    time_t now = time(NULL);
    
    switch (cb->state) {
        case OPEN:
            if (now - cb->last_failure_time > cb->timeout_seconds) {
                cb->state = HALF_OPEN;
                break;
            }
            return -1;  // Circuit is open
            
        case HALF_OPEN:
            // Try one request
            break;
            
        case CLOSED:
            // Normal operation
            break;
    }
    
    int result = operation(arg);
    
    if (result == 0) {
        // Success
        cb->failure_count = 0;
        cb->state = CLOSED;
    } else {
        // Failure
        cb->failure_count++;
        cb->last_failure_time = now;
        
        if (cb->failure_count >= cb->failure_threshold) {
            cb->state = OPEN;
        }
    }
    
    return result;
}
```

### Performance Optimization Patterns
```c
// Pattern 1: Batch Operations
int batch_send(int sockfd, const void** buffers, size_t* lengths, int count) {
    struct iovec iov[count];
    
    for (int i = 0; i < count; i++) {
        iov[i].iov_base = (void*)buffers[i];
        iov[i].iov_len = lengths[i];
    }
    
    struct msghdr msg = {0};
    msg.msg_iov = iov;
    msg.msg_iovlen = count;
    
    return sendmsg(sockfd, &msg, 0);
}

// Pattern 2: Asynchronous I/O with epoll
int async_client_loop(int* sockfds, int count) {
    int epollfd = epoll_create1(0);
    if (epollfd == -1) {
        return -1;
    }
    
    // Add sockets to epoll
    for (int i = 0; i < count; i++) {
        struct epoll_event ev;
        ev.events = EPOLLIN | EPOLLOUT | EPOLLET;  // Edge-triggered
        ev.data.fd = sockfds[i];
        epoll_ctl(epollfd, EPOLL_CTL_ADD, sockfds[i], &ev);
    }
    
    struct epoll_event events[10];
    
    while (1) {
        int nfds = epoll_wait(epollfd, events, 10, -1);
        
        for (int i = 0; i < nfds; i++) {
            int fd = events[i].data.fd;
            
            if (events[i].events & EPOLLIN) {
                handle_read_ready(fd);
            }
            
            if (events[i].events & EPOLLOUT) {
                handle_write_ready(fd);
            }
            
            if (events[i].events & (EPOLLHUP | EPOLLERR)) {
                handle_error(fd);
            }
        }
    }
    
    close(epollfd);
    return 0;
}
```

## Next Steps

After mastering TCP client programming:
- Proceed to TCP Server Programming
- Explore advanced client patterns (connection pooling, reconnection strategies)
- Learn about non-blocking client implementations

## Resources and Further Learning

### Essential Reading
- **"UNIX Network Programming, Volume 1" by W. Richard Stevens** (Chapters 4-6)
  - The definitive guide to socket programming
  - Covers client-server fundamentals in depth
  - Essential for understanding low-level networking

- **"TCP/IP Illustrated, Volume 1" by W. Richard Stevens**
  - Understanding the TCP protocol itself
  - How TCP handles reliability, flow control, and congestion control
  - Essential for troubleshooting network issues

- **"Beej's Guide to Network Programming"** - [Online Free Resource](https://beej.us/guide/bgnet/)
  - Practical, example-driven approach
  - Great for beginners
  - Covers both IPv4 and IPv6

### Online Resources

#### Documentation
- **Linux Man Pages**: `man 2 socket`, `man 2 connect`, `man 2 send`, `man 2 recv`
- **POSIX.1-2017 Standard**: Official specification for socket API
- **RFC 793**: TCP Protocol Specification
- **RFC 1122**: Requirements for Internet Hosts

#### Tutorials and Guides
- [IBM Developer - Socket Programming](https://developer.ibm.com/technologies/systems/tutorials/l-sock/)
- [Oracle Socket Programming Tutorial](https://docs.oracle.com/cd/E19455-01/806-1017/6ja85399b/index.html)
- [GeeksforGeeks Network Programming](https://www.geeksforgeeks.org/socket-programming-cc/)

#### Tools Documentation
- **Wireshark User Guide**: For packet capture and analysis
- **netstat/ss Manual**: For connection monitoring
- **tcpdump Manual**: For command-line packet capture

### Development Tools

#### Debugging and Analysis
```bash
# Network connectivity testing
ping <hostname>
telnet <hostname> <port>
nc -zv <hostname> <port>

# Socket monitoring
netstat -tulpn
ss -tulpn
lsof -i :<port>

# Packet capture
tcpdump -i any -n port <port>
wireshark

# Performance analysis
iperf3 -c <server> -p <port>
nload
iftop
```

#### Development Environment
```bash
# Compiler flags for debugging
gcc -g -Wall -Wextra -pthread -o client client.c

# Memory debugging
valgrind --tool=memcheck ./client
valgrind --tool=helgrind ./client  # For threading issues

# Static analysis
cppcheck client.c
clang-static-analyzer client.c
```

### Code Libraries and Frameworks

#### C/C++ Libraries
- **libevent**: Event-driven programming for scalable network servers
- **libev**: High-performance event loop library
- **boost::asio**: C++ async I/O library
- **OpenSSL**: For secure socket implementations
- **c-ares**: Asynchronous DNS resolution

#### Protocol Libraries
- **libcurl**: HTTP/HTTPS client library
- **libzmq**: High-level messaging library
- **protobuf-c**: Protocol Buffers for C
- **jansson**: JSON library for C

### Practice Servers for Testing

#### Public Echo Servers
```bash
# Test your client against these
telnet echo.moria.us 4321
telnet tcpbin.com 4242
```

#### Setting Up Test Servers
```bash
# Simple netcat server
nc -l -p 8080

# Python echo server
python3 -c "
import socket
s = socket.socket()
s.bind(('localhost', 8080))
s.listen(5)
print('Server listening on port 8080')
while True:
    c, a = s.accept()
    data = c.recv(1024)
    c.send(data)
    c.close()
"

# HTTP test server
python3 -m http.server 8080
```

### Advanced Topics to Explore Next

#### After mastering basic TCP client programming:

1. **TCP Server Programming**
   - Accept connections and handle multiple clients
   - Implement concurrent server patterns
   - Load balancing and connection management

2. **Advanced Socket Options**
   - TCP_FASTOPEN for reduced latency
   - SO_ZEROCOPY for high-performance applications
   - Socket priorities and QoS

3. **Secure Communications**
   - SSL/TLS implementation
   - Certificate validation
   - Secure key exchange

4. **High-Performance Networking**
   - Non-blocking I/O and event loops
   - io_uring on Linux
   - DPDK for kernel bypass

5. **Network Programming Patterns**
   - Reactor and Proactor patterns
   - Connection pooling strategies
   - Circuit breakers and retry policies

6. **Protocol Design**
   - Binary vs text protocols
   - Message framing and serialization
   - Version compatibility

### Certification and Career Development

#### Industry Certifications
- **Linux Professional Institute (LPI)**: Linux networking concepts
- **Red Hat Certified Engineer (RHCE)**: System administration with networking
- **Cisco CCNA**: Networking fundamentals (though focused on routing/switching)

#### Career Paths
- **Systems Programmer**: Low-level network application development
- **Network Software Engineer**: Building networking infrastructure
- **DevOps Engineer**: Network automation and monitoring
- **Security Engineer**: Network security and penetration testing
- **Embedded Systems Developer**: IoT and embedded networking

### Contributing to Open Source

#### Projects to Contribute To
- **curl**: HTTP client library and command-line tool
- **nginx**: High-performance web server
- **HAProxy**: Load balancer and proxy server
- **Redis**: In-memory data store with networking components
- **PostgreSQL**: Database with network protocol implementation

#### How to Contribute
1. Start with documentation improvements
2. Fix simple bugs marked as "good first issue"
3. Add test cases for existing functionality
4. Implement small feature requests
5. Optimize performance in networking code

This comprehensive resource list should provide you with everything needed to master TCP client programming and advance to more complex networking topics.

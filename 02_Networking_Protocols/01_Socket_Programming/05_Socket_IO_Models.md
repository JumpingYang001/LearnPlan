# Socket I/O Models

*Last Updated: June 21, 2025*

## Overview

Socket I/O models determine how applications handle input/output operations with network sockets. The choice of I/O model significantly impacts application performance, scalability, and complexity. This module provides comprehensive coverage of different I/O models, from simple blocking approaches to advanced event-driven architectures.

### Why I/O Models Matter

**The Fundamental Problem:**
When a network application needs to handle multiple clients simultaneously, the I/O model determines:
- How many connections can be handled concurrently
- How CPU and memory resources are utilized
- Application responsiveness and throughput
- Code complexity and maintainability

**Historical Context:**
```
Traditional Approach (1 thread per connection):
┌─────────┐    ┌─────────┐    ┌─────────┐
│Thread 1 │    │Thread 2 │    │Thread N │
│Client A │    │Client B │    │Client N │
└─────────┘    └─────────┘    └─────────┘
     │              │              │
     └──────────────┼──────────────┘
                    │
              ┌─────────────┐
              │   Server    │
              └─────────────┘

Problems:
- Thread creation overhead
- Memory usage (8MB stack per thread)
- Context switching costs
- Maximum threads limited by system resources
```

**Modern Approach (Event-driven):**
```
Event-Driven Architecture:
┌─────────────────────────────────────┐
│          Event Loop                 │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐│
│  │Evt 1│  │Evt 2│  │Evt 3│  │Evt N││
│  └─────┘  └─────┘  └─────┘  └─────┘│
└─────────────────────────────────────┘
             │
    ┌────────┼────────┐
    │        │        │
┌───▼───┐ ┌──▼──┐ ┌───▼───┐
│Client │ │Client│ │Client │
│   A   │ │  B  │ │   N   │
└───────┘ └─────┘ └───────┘

Benefits:
- Single thread handles multiple connections
- Lower memory footprint
- No context switching overhead
- Scales to thousands of connections
```

### Performance Comparison Overview

| Connections | Blocking I/O | Non-blocking + select() | Epoll/Kqueue |
|-------------|--------------|-------------------------|--------------|
| 10          | ✅ Good       | ✅ Good                  | ⚠️ Overkill   |
| 100         | ⚠️ Degraded   | ✅ Good                  | ✅ Excellent  |
| 1,000       | ❌ Poor       | ⚠️ Degraded              | ✅ Excellent  |
| 10,000+     | ❌ Impossible | ❌ Poor                  | ✅ Excellent  |

## Learning Objectives

By the end of this module, you should be able to:
- **Understand and compare** different I/O models and their performance characteristics
- **Implement blocking I/O** with proper timeout handling and error management
- **Master non-blocking I/O** patterns and handle asynchronous operations correctly
- **Use I/O multiplexing** effectively with select(), poll(), and modern alternatives
- **Design event-driven applications** using epoll (Linux), kqueue (BSD/macOS), or IOCP (Windows)
- **Choose appropriate I/O models** based on application requirements and system constraints
- **Optimize network applications** for high concurrency and low latency
- **Debug and profile** I/O-intensive applications using system tools

### Self-Assessment Checklist

Before proceeding, ensure you can:

□ Explain the difference between synchronous/asynchronous and blocking/non-blocking I/O  
□ Implement a multi-client server using each I/O model  
□ Handle edge cases like partial reads, connection errors, and resource exhaustion  
□ Measure and compare performance of different I/O models  
□ Choose the right I/O model for specific application requirements  
□ Debug I/O-related issues using system monitoring tools  
□ Implement proper error handling and resource cleanup  

### Practical Competencies

After completing this module, you should be able to build:
- **Chat servers** handling hundreds of concurrent users
- **HTTP servers** with high request throughput
- **Proxy servers** with low latency forwarding
- **Game servers** with real-time communication
- **IoT gateways** managing thousands of device connections

## Topics Covered

## Topics Covered

### 1. Blocking I/O (Synchronous I/O)

#### Understanding Blocking Behavior

Blocking I/O means that when a thread calls a function like `recv()`, `send()`, or `accept()`, the thread **blocks** (sleeps) until the operation can complete. This is the simplest model but has significant limitations for concurrent applications.

**How Blocking I/O Works:**
```
Timeline for recv() call on empty socket:

Thread calls recv()
     │
     ▼
┌─────────────────────────────┐
│   Thread BLOCKED/SLEEPING   │  ← CPU does other work
│   Waiting for data...       │
└─────────────────────────────┘
                │
    Data arrives from network
                │
                ▼
     recv() returns with data
                │
                ▼
     Thread continues execution
```

#### Synchronous Communication Patterns

**Simple Request-Response Pattern:**
```c
// Server side - handles one client at a time
void handle_client_blocking(int client_fd) {
    char buffer[1024];
    
    while (1) {
        // BLOCKS until data arrives
        ssize_t bytes_received = recv(client_fd, buffer, sizeof(buffer), 0);
        
        if (bytes_received <= 0) {
            break; // Client disconnected or error
        }
        
        // Process request
        process_request(buffer, bytes_received);
        
        // BLOCKS until data is sent
        send(client_fd, response, response_len, 0);
    }
    
    close(client_fd);
}
```

**Multi-threaded Blocking Server:**
```c
#include <pthread.h>

typedef struct {
    int client_fd;
    struct sockaddr_in client_addr;
} client_info_t;

void* client_handler_thread(void* arg) {
    client_info_t* client = (client_info_t*)arg;
    
    printf("Handling client %s:%d\n", 
           inet_ntoa(client->client_addr.sin_addr),
           ntohs(client->client_addr.sin_port));
    
    handle_client_blocking(client->client_fd);
    
    free(client);
    return NULL;
}

void blocking_server_with_threads(int server_fd) {
    while (1) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        
        // BLOCKS until new connection arrives
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addr_len);
        
        if (client_fd < 0) {
            perror("accept");
            continue;
        }
        
        // Create new thread for each client
        client_info_t* client = malloc(sizeof(client_info_t));
        client->client_fd = client_fd;
        client->client_addr = client_addr;
        
        pthread_t thread;
        if (pthread_create(&thread, NULL, client_handler_thread, client) != 0) {
            perror("pthread_create");
            close(client_fd);
            free(client);
        } else {
            pthread_detach(thread); // Don't need to join
        }
    }
}
```

#### Timeout Handling Mechanisms

**Problem with Basic Blocking I/O:**
```c
// This can block FOREVER if client doesn't send data
ssize_t bytes = recv(client_fd, buffer, sizeof(buffer), 0);
```

**Solution 1: Socket-level Timeouts**
```c
#include <sys/socket.h>

int set_socket_timeout(int sockfd, int timeout_seconds) {
    struct timeval timeout;
    timeout.tv_sec = timeout_seconds;
    timeout.tv_usec = 0;
    
    // Set receive timeout
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, 
                   &timeout, sizeof(timeout)) < 0) {
        perror("setsockopt SO_RCVTIMEO");
        return -1;
    }
    
    // Set send timeout
    if (setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, 
                   &timeout, sizeof(timeout)) < 0) {
        perror("setsockopt SO_SNDTIMEO");
        return -1;
    }
    
    return 0;
}

// Usage example
void client_with_timeout(int client_fd) {
    set_socket_timeout(client_fd, 30); // 30 second timeout
    
    char buffer[1024];
    ssize_t bytes = recv(client_fd, buffer, sizeof(buffer), 0);
    
    if (bytes < 0) {
        if (errno == EWOULDBLOCK || errno == EAGAIN) {
            printf("Timeout: Client didn't send data within 30 seconds\n");
        } else {
            perror("recv error");
        }
    }
}
```

**Solution 2: Using select() for Timeout Control**
```c
int recv_with_timeout(int sockfd, void* buffer, size_t len, int timeout_sec) {
    fd_set readfds;
    struct timeval timeout;
    
    FD_ZERO(&readfds);
    FD_SET(sockfd, &readfds);
    
    timeout.tv_sec = timeout_sec;
    timeout.tv_usec = 0;
    
    // Wait for socket to become readable
    int result = select(sockfd + 1, &readfds, NULL, NULL, &timeout);
    
    if (result < 0) {
        perror("select failed");
        return -1;
    } else if (result == 0) {
        // Timeout occurred
        errno = ETIMEDOUT;
        return -1;
    }
    
    // Socket is ready, now we can read without blocking
    return recv(sockfd, buffer, len, 0);
}
```

#### Advantages and Limitations

**✅ Advantages:**
- **Simple Programming Model**: Easy to understand and implement
- **Sequential Logic**: Code flows naturally from top to bottom
- **Debugging Friendly**: Easy to trace execution flow
- **Resource Predictability**: Each thread has predictable resource usage

**❌ Limitations:**
- **Poor Scalability**: One thread per connection doesn't scale beyond ~1000 connections
- **Resource Intensive**: Each thread uses ~8MB of stack space
- **Context Switching Overhead**: OS must switch between thousands of threads
- **Thread Management Complexity**: Creating, destroying, and coordinating threads
- **Deadlock Potential**: Multiple threads accessing shared resources

**When to Use Blocking I/O:**
```c
// Good for:
// 1. Simple client applications
int connect_and_download(const char* server, int port, const char* file) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    // ... connect to server ...
    
    // Simple request-response - blocking is fine here
    send(sockfd, request, strlen(request), 0);
    recv(sockfd, response, sizeof(response), 0);
    
    return 0;
}

// 2. Low-concurrency servers (< 50 clients)
// 3. Applications where simplicity is more important than performance
// 4. Prototype development
```

### 2. Non-blocking I/O (Asynchronous I/O)

#### Understanding Non-blocking Behavior

Non-blocking I/O means that I/O operations return immediately, even if they cannot complete. If data isn't available, the function returns with an error code (`EAGAIN` or `EWOULDBLOCK`) instead of waiting.

**How Non-blocking I/O Works:**
```
Timeline for recv() call on empty socket:

Thread calls recv()
     │
     ▼
recv() checks socket buffer
     │
     ▼
No data available
     │
     ▼
recv() returns -1 (EAGAIN)
     │
     ▼
Thread continues immediately
     │
     ▼
Thread can do other work or try again later
```

#### Setting Non-blocking Mode

**Method 1: Using fcntl() (POSIX)**
```c
#include <fcntl.h>

int make_socket_non_blocking(int sockfd) {
    int flags = fcntl(sockfd, F_GETFL, 0);
    if (flags < 0) {
        perror("fcntl F_GETFL");
        return -1;
    }
    
    flags |= O_NONBLOCK;
    if (fcntl(sockfd, F_SETFL, flags) < 0) {
        perror("fcntl F_SETFL");
        return -1;
    }
    
    return 0;
}
```

**Method 2: Using ioctl() (Alternative)**
```c
#include <sys/ioctl.h>

int make_socket_non_blocking_ioctl(int sockfd) {
    int on = 1;
    if (ioctl(sockfd, FIONBIO, &on) < 0) {
        perror("ioctl FIONBIO");
        return -1;
    }
    return 0;
}
```

#### Polling with Non-blocking Sockets

**Basic Polling Pattern:**
```c
ssize_t non_blocking_recv_with_retry(int sockfd, void* buffer, size_t length, int max_attempts) {
    for (int attempt = 0; attempt < max_attempts; attempt++) {
        ssize_t result = recv(sockfd, buffer, length, 0);
        
        if (result >= 0) {
            return result; // Success or connection closed
        }
        
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // No data available, try again after a short delay
            usleep(1000); // Sleep for 1ms
            continue;
        } else {
            // Real error
            perror("recv");
            return -1;
        }
    }
    
    // Maximum attempts reached
    errno = ETIMEDOUT;
    return -1;
}
```

#### Handling EAGAIN/EWOULDBLOCK

**Comprehensive Error Handling:**
```c
typedef enum {
    IO_SUCCESS,
    IO_WOULD_BLOCK,
    IO_ERROR,
    IO_CONNECTION_CLOSED
} io_result_t;

io_result_t non_blocking_send_all(int sockfd, const void* data, size_t length, size_t* bytes_sent) {
    const char* ptr = (const char*)data;
    size_t total_sent = 0;
    
    while (total_sent < length) {
        ssize_t sent = send(sockfd, ptr + total_sent, length - total_sent, 0);
        
        if (sent > 0) {
            total_sent += sent;
        } else if (sent == 0) {
            // This shouldn't happen with send(), but handle it
            break;
        } else {
            // sent < 0, check errno
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Socket buffer is full, can't send more right now
                *bytes_sent = total_sent;
                return IO_WOULD_BLOCK;
            } else if (errno == EINTR) {
                // Interrupted by signal, retry
                continue;
            } else if (errno == EPIPE || errno == ECONNRESET) {
                // Connection broken
                *bytes_sent = total_sent;
                return IO_CONNECTION_CLOSED;
            } else {
                // Other error
                perror("send");
                *bytes_sent = total_sent;
                return IO_ERROR;
            }
        }
    }
    
    *bytes_sent = total_sent;
    return IO_SUCCESS;
}

io_result_t non_blocking_recv_some(int sockfd, void* buffer, size_t length, size_t* bytes_received) {
    ssize_t received = recv(sockfd, buffer, length, 0);
    
    if (received > 0) {
        *bytes_received = received;
        return IO_SUCCESS;
    } else if (received == 0) {
        // Connection closed by peer
        *bytes_received = 0;
        return IO_CONNECTION_CLOSED;
    } else {
        // received < 0, check errno
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // No data available right now
            *bytes_received = 0;
            return IO_WOULD_BLOCK;
        } else if (errno == EINTR) {
            // Interrupted, try again
            return non_blocking_recv_some(sockfd, buffer, length, bytes_received);
        } else {
            // Real error
            perror("recv");
            *bytes_received = 0;
            return IO_ERROR;
        }
    }
}
```

**Practical Non-blocking Server Example:**
```c
#include <sys/socket.h>
#include <fcntl.h>
#include <errno.h>

#define MAX_CLIENTS 1000

typedef struct {
    int fd;
    char recv_buffer[4096];
    size_t recv_len;
    char send_buffer[4096];
    size_t send_len;
    size_t send_pos;
} client_connection_t;

void non_blocking_server(int server_fd) {
    client_connection_t clients[MAX_CLIENTS];
    int client_count = 0;
    
    // Make server socket non-blocking
    make_socket_non_blocking(server_fd);
    
    // Initialize client array
    for (int i = 0; i < MAX_CLIENTS; i++) {
        clients[i].fd = -1;
    }
    
    while (1) {
        // Try to accept new connections
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        
        int new_client = accept(server_fd, (struct sockaddr*)&client_addr, &addr_len);
        if (new_client >= 0) {
            // Find free slot
            int slot = -1;
            for (int i = 0; i < MAX_CLIENTS; i++) {
                if (clients[i].fd == -1) {
                    slot = i;
                    break;
                }
            }
            
            if (slot != -1) {
                make_socket_non_blocking(new_client);
                clients[slot].fd = new_client;
                clients[slot].recv_len = 0;
                clients[slot].send_len = 0;
                clients[slot].send_pos = 0;
                client_count++;
                
                printf("New client connected (total: %d)\n", client_count);
            } else {
                printf("Max clients reached, rejecting connection\n");
                close(new_client);
            }
        } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
            perror("accept");
        }
        
        // Process existing clients
        for (int i = 0; i < MAX_CLIENTS; i++) {
            if (clients[i].fd == -1) continue;
            
            client_connection_t* client = &clients[i];
            
            // Try to read from client
            size_t bytes_received;
            io_result_t recv_result = non_blocking_recv_some(client->fd, 
                                                           client->recv_buffer + client->recv_len,
                                                           sizeof(client->recv_buffer) - client->recv_len,
                                                           &bytes_received);
            
            if (recv_result == IO_SUCCESS) {
                client->recv_len += bytes_received;
                
                // Process complete messages (assuming line-based protocol)
                char* line_end = memchr(client->recv_buffer, '\n', client->recv_len);
                if (line_end) {
                    size_t line_len = line_end - client->recv_buffer + 1;
                    
                    // Echo the line back
                    memcpy(client->send_buffer, client->recv_buffer, line_len);
                    client->send_len = line_len;
                    client->send_pos = 0;
                    
                    // Remove processed data from buffer
                    memmove(client->recv_buffer, line_end + 1, client->recv_len - line_len);
                    client->recv_len -= line_len;
                }
            } else if (recv_result == IO_CONNECTION_CLOSED || recv_result == IO_ERROR) {
                printf("Client disconnected\n");
                close(client->fd);
                client->fd = -1;
                client_count--;
            }
            
            // Try to send pending data
            if (client->send_len > client->send_pos) {
                size_t bytes_sent;
                io_result_t send_result = non_blocking_send_all(client->fd,
                                                              client->send_buffer + client->send_pos,
                                                              client->send_len - client->send_pos,
                                                              &bytes_sent);
                
                client->send_pos += bytes_sent;
                
                if (client->send_pos >= client->send_len) {
                    // All data sent
                    client->send_len = 0;
                    client->send_pos = 0;
                }
                
                if (send_result == IO_CONNECTION_CLOSED || send_result == IO_ERROR) {
                    printf("Client send error, disconnecting\n");
                    close(client->fd);
                    client->fd = -1;
                    client_count--;
                }
            }
        }
        
        // Small delay to prevent busy-waiting
        usleep(1000); // 1ms
    }
}
```

### 3. I/O Multiplexing

I/O Multiplexing allows a single thread to monitor multiple file descriptors (sockets) simultaneously and respond when any of them are ready for I/O operations. This is the foundation of high-performance network servers.

#### Core Concept: The Event Loop

```
Classic Problem: How to handle multiple clients with one thread?

Without I/O Multiplexing:
┌─────────────────────────────────────┐
│ while (1) {                         │
│   client1_data = recv(client1, ...); │ ← Blocks here
│   client2_data = recv(client2, ...); │ ← Can't reach if client1 blocks
│   client3_data = recv(client3, ...); │ ← Can't reach if client2 blocks
│ }                                   │
└─────────────────────────────────────┘

With I/O Multiplexing:
┌─────────────────────────────────────┐
│ while (1) {                         │
│   ready_sockets = select(all_sockets); │ ← Monitor all sockets
│   for (each ready_socket) {         │
│     data = recv(ready_socket, ...); │ ← Only read from ready sockets
│     process(data);                  │
│   }                                 │
│ }                                   │
└─────────────────────────────────────┘
```

#### select() System Call

The `select()` system call is the oldest and most portable I/O multiplexing mechanism.

**How select() Works:**
```c
int select(int nfds,                    // Highest fd + 1
          fd_set *readfds,             // Sockets to monitor for reading
          fd_set *writefds,            // Sockets to monitor for writing  
          fd_set *exceptfds,           // Sockets to monitor for exceptions
          struct timeval *timeout);    // Timeout or NULL for blocking

// Returns: number of ready descriptors, 0 on timeout, -1 on error
```

**Understanding fd_set:**
```c
// fd_set is a bit array where each bit represents a file descriptor
// Maximum FD_SETSIZE file descriptors (typically 1024)

fd_set readfds;
FD_ZERO(&readfds);        // Clear all bits
FD_SET(sockfd, &readfds); // Set bit for sockfd
FD_CLR(sockfd, &readfds); // Clear bit for sockfd
FD_ISSET(sockfd, &readfds); // Test if bit is set

// Visual representation:
// fd_set: [0][1][0][1][0][0][1][0] ...
//          ^   ^       ^       ^
//         fd0 fd1     fd3     fd6 are set
```

**Comprehensive select() Server Example:**
```c
#include <sys/select.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define MAX_CLIENTS 50
#define BUFFER_SIZE 1024

typedef struct {
    int fd;
    struct sockaddr_in addr;
    char buffer[BUFFER_SIZE];
    size_t buffer_len;
    time_t last_activity;
} client_info_t;

void select_based_echo_server(int server_fd) {
    client_info_t clients[MAX_CLIENTS];
    fd_set master_readfds, master_writefds;
    int max_fd = server_fd;
    int client_count = 0;
    
    // Initialize
    FD_ZERO(&master_readfds);
    FD_ZERO(&master_writefds);
    FD_SET(server_fd, &master_readfds);
    
    for (int i = 0; i < MAX_CLIENTS; i++) {
        clients[i].fd = -1;
    }
    
    printf("Select-based server listening...\n");
    
    while (1) {
        fd_set read_fds = master_readfds;
        fd_set write_fds = master_writefds;
        
        // Set timeout to handle periodic tasks
        struct timeval timeout;
        timeout.tv_sec = 1;  // 1 second timeout
        timeout.tv_usec = 0;
        
        int activity = select(max_fd + 1, &read_fds, &write_fds, NULL, &timeout);
        
        if (activity < 0) {
            perror("select failed");
            break;
        } else if (activity == 0) {
            // Timeout - do periodic maintenance
            time_t now = time(NULL);
            printf("Server heartbeat: %d clients connected\n", client_count);
            
            // Check for idle clients (timeout after 60 seconds)
            for (int i = 0; i < MAX_CLIENTS; i++) {
                if (clients[i].fd != -1 && (now - clients[i].last_activity) > 60) {
                    printf("Closing idle client\n");
                    close(clients[i].fd);
                    FD_CLR(clients[i].fd, &master_readfds);
                    FD_CLR(clients[i].fd, &master_writefds);
                    clients[i].fd = -1;
                    client_count--;
                }
            }
            continue;
        }
        
        // Check for new connections
        if (FD_ISSET(server_fd, &read_fds)) {
            struct sockaddr_in client_addr;
            socklen_t addr_len = sizeof(client_addr);
            
            int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addr_len);
            if (client_fd >= 0) {
                // Find free slot
                int slot = -1;
                for (int i = 0; i < MAX_CLIENTS; i++) {
                    if (clients[i].fd == -1) {
                        slot = i;
                        break;
                    }
                }
                
                if (slot != -1 && client_count < MAX_CLIENTS) {
                    clients[slot].fd = client_fd;
                    clients[slot].addr = client_addr;
                    clients[slot].buffer_len = 0;
                    clients[slot].last_activity = time(NULL);
                    
                    FD_SET(client_fd, &master_readfds);
                    if (client_fd > max_fd) {
                        max_fd = client_fd;
                    }
                    
                    client_count++;
                    printf("New client connected: %s:%d (total: %d)\n",
                           inet_ntoa(client_addr.sin_addr),
                           ntohs(client_addr.sin_port),
                           client_count);
                } else {
                    printf("Maximum clients reached, rejecting connection\n");
                    close(client_fd);
                }
            }
        }
        
        // Check existing clients for data
        for (int i = 0; i < MAX_CLIENTS; i++) {
            if (clients[i].fd == -1) continue;
            
            int client_fd = clients[i].fd;
            
            // Check for incoming data
            if (FD_ISSET(client_fd, &read_fds)) {
                char temp_buffer[BUFFER_SIZE];
                ssize_t bytes_read = recv(client_fd, temp_buffer, sizeof(temp_buffer) - 1, 0);
                
                if (bytes_read <= 0) {
                    // Client disconnected or error
                    if (bytes_read == 0) {
                        printf("Client disconnected gracefully\n");
                    } else {
                        perror("recv error");
                    }
                    
                    close(client_fd);
                    FD_CLR(client_fd, &master_readfds);
                    FD_CLR(client_fd, &master_writefds);
                    clients[i].fd = -1;
                    client_count--;
                } else {
                    // Data received - prepare echo
                    temp_buffer[bytes_read] = '\0';
                    printf("Received from client: %s", temp_buffer);
                    
                    // Copy to client buffer for echoing
                    if (bytes_read < BUFFER_SIZE) {
                        memcpy(clients[i].buffer, temp_buffer, bytes_read);
                        clients[i].buffer_len = bytes_read;
                        FD_SET(client_fd, &master_writefds); // Mark for writing
                    }
                    
                    clients[i].last_activity = time(NULL);
                }
            }
            
            // Check for outgoing data
            if (FD_ISSET(client_fd, &write_fds) && clients[i].buffer_len > 0) {
                ssize_t bytes_sent = send(client_fd, clients[i].buffer, 
                                        clients[i].buffer_len, 0);
                
                if (bytes_sent > 0) {
                    printf("Echoed %zd bytes back to client\n", bytes_sent);
                    clients[i].buffer_len = 0;
                    FD_CLR(client_fd, &master_writefds); // No more data to send
                } else if (bytes_sent < 0 && errno != EAGAIN) {
                    perror("send error");
                    close(client_fd);
                    FD_CLR(client_fd, &master_readfds);
                    FD_CLR(client_fd, &master_writefds);
                    clients[i].fd = -1;
                    client_count--;
                }
            }
        }
        
        // Update max_fd if needed
        if (max_fd >= FD_SETSIZE) {
            max_fd = server_fd;
            for (int i = 0; i < MAX_CLIENTS; i++) {
                if (clients[i].fd > max_fd) {
                    max_fd = clients[i].fd;
                }
            }
        }
    }
}
```

#### poll() System Call

`poll()` is a more modern alternative to `select()` that doesn't have the FD_SETSIZE limitation and provides a cleaner interface.

**How poll() Works:**
```c
#include <poll.h>

struct pollfd {
    int fd;          // File descriptor to monitor
    short events;    // Events to monitor (POLLIN, POLLOUT, etc.)
    short revents;   // Events that occurred (filled by poll())
};

int poll(struct pollfd *fds,    // Array of file descriptors
         nfds_t nfds,          // Number of descriptors
         int timeout);         // Timeout in milliseconds (-1 = infinite)
```

**poll() Event Types:**
```c
// Input events (events field)
POLLIN     // Data available for reading
POLLOUT    // Socket ready for writing
POLLPRI    // Urgent data available

// Output events (revents field - set by poll())
POLLIN     // Data available
POLLOUT    // Can write without blocking
POLLERR    // Error condition
POLLHUP    // Hang up (connection closed)
POLLNVAL   // Invalid file descriptor
```

**Comprehensive poll() Server Example:**
```c
#include <poll.h>

#define MAX_CLIENTS 1000

typedef struct {
    char buffer[1024];
    size_t buffer_len;
    time_t last_activity;
} client_data_t;

void poll_based_server(int server_fd) {
    struct pollfd fds[MAX_CLIENTS];
    client_data_t client_data[MAX_CLIENTS];
    int nfds = 1;
    
    // Initialize with server socket
    fds[0].fd = server_fd;
    fds[0].events = POLLIN;
    
    // Initialize client slots
    for (int i = 1; i < MAX_CLIENTS; i++) {
        fds[i].fd = -1;
        client_data[i].buffer_len = 0;
    }
    
    printf("Poll-based server listening...\n");
    
    while (1) {
        // Poll with 1 second timeout
        int poll_count = poll(fds, nfds, 1000);
        
        if (poll_count < 0) {
            perror("Poll failed");
            break;
        } else if (poll_count == 0) {
            // Timeout - do maintenance
            printf("Server heartbeat: %d clients\n", nfds - 1);
            continue;
        }
        
        // Check server socket for new connections
        if (fds[0].revents & POLLIN) {
            struct sockaddr_in client_addr;
            socklen_t addr_len = sizeof(client_addr);
            
            int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addr_len);
            if (client_fd >= 0) {
                if (nfds < MAX_CLIENTS) {
                    fds[nfds].fd = client_fd;
                    fds[nfds].events = POLLIN;
                    client_data[nfds].buffer_len = 0;
                    client_data[nfds].last_activity = time(NULL);
                    nfds++;
                    
                    printf("New client connected, total clients: %d\n", nfds - 1);
                } else {
                    printf("Maximum clients reached\n");
                    close(client_fd);
                }
            }
        }
        
        // Check client sockets
        for (int i = 1; i < nfds; i++) {
            if (fds[i].fd == -1) continue;
            
            // Handle errors first
            if (fds[i].revents & (POLLERR | POLLHUP | POLLNVAL)) {
                printf("Client error/hangup on fd %d\n", fds[i].fd);
                close(fds[i].fd);
                
                // Remove from array by moving last element here
                fds[i] = fds[nfds - 1];
                client_data[i] = client_data[nfds - 1];
                nfds--;
                i--; // Recheck this position
                continue;
            }
            
            // Handle incoming data
            if (fds[i].revents & POLLIN) {
                char buffer[1024];
                ssize_t bytes_read = recv(fds[i].fd, buffer, sizeof(buffer), 0);
                
                if (bytes_read <= 0) {
                    // Client disconnected
                    printf("Client disconnected\n");
                    close(fds[i].fd);
                    
                    // Remove from array
                    fds[i] = fds[nfds - 1];
                    client_data[i] = client_data[nfds - 1];
                    nfds--;
                    i--;
                } else {
                    // Echo data back
                    send(fds[i].fd, buffer, bytes_read, 0);
                    client_data[i].last_activity = time(NULL);
                }
            }
        }
    }
}
```

#### Implementation Patterns and Best Practices

**Pattern 1: State Machine for Complex Protocols**
```c
typedef enum {
    CLIENT_STATE_READING_HEADER,
    CLIENT_STATE_READING_BODY,
    CLIENT_STATE_PROCESSING,
    CLIENT_STATE_SENDING_RESPONSE
} client_state_t;

typedef struct {
    int fd;
    client_state_t state;
    char header_buffer[64];
    size_t header_pos;
    char* body_buffer;
    size_t body_size;
    size_t body_pos;
    char* response_buffer;
    size_t response_size;
    size_t response_pos;
} stateful_client_t;

void handle_client_state_machine(stateful_client_t* client, short revents) {
    switch (client->state) {
        case CLIENT_STATE_READING_HEADER:
            if (revents & POLLIN) {
                // Read header until we have complete header
                ssize_t bytes = recv(client->fd, 
                                   client->header_buffer + client->header_pos,
                                   sizeof(client->header_buffer) - client->header_pos,
                                   0);
                if (bytes > 0) {
                    client->header_pos += bytes;
                    
                    // Check if header is complete (e.g., ends with \r\n\r\n)
                    if (strstr(client->header_buffer, "\r\n\r\n")) {
                        // Parse header to determine body size
                        client->body_size = parse_content_length(client->header_buffer);
                        client->body_buffer = malloc(client->body_size);
                        client->body_pos = 0;
                        client->state = CLIENT_STATE_READING_BODY;
                    }
                }
            }
            break;
            
        case CLIENT_STATE_READING_BODY:
            if (revents & POLLIN) {
                ssize_t bytes = recv(client->fd,
                                   client->body_buffer + client->body_pos,
                                   client->body_size - client->body_pos,
                                   0);
                if (bytes > 0) {
                    client->body_pos += bytes;
                    
                    if (client->body_pos >= client->body_size) {
                        client->state = CLIENT_STATE_PROCESSING;
                    }
                }
            }
            break;
            
        case CLIENT_STATE_PROCESSING:
            // Process request and prepare response
            client->response_buffer = process_request(client->header_buffer, 
                                                    client->body_buffer,
                                                    &client->response_size);
            client->response_pos = 0;
            client->state = CLIENT_STATE_SENDING_RESPONSE;
            break;
            
        case CLIENT_STATE_SENDING_RESPONSE:
            if (revents & POLLOUT) {
                ssize_t bytes = send(client->fd,
                                   client->response_buffer + client->response_pos,
                                   client->response_size - client->response_pos,
                                   0);
                if (bytes > 0) {
                    client->response_pos += bytes;
                    
                    if (client->response_pos >= client->response_size) {
                        // Response sent completely, reset for next request
                        free(client->body_buffer);
                        free(client->response_buffer);
                        client->state = CLIENT_STATE_READING_HEADER;
                        client->header_pos = 0;
                    }
                }
            }
            break;
    }
}
```

**Pattern 2: Buffer Management**
```c
typedef struct {
    char* data;
    size_t capacity;
    size_t length;
    size_t read_pos;
} circular_buffer_t;

circular_buffer_t* create_buffer(size_t capacity) {
    circular_buffer_t* buf = malloc(sizeof(circular_buffer_t));
    buf->data = malloc(capacity);
    buf->capacity = capacity;
    buf->length = 0;
    buf->read_pos = 0;
    return buf;
}

int buffer_write(circular_buffer_t* buf, const void* data, size_t len) {
    if (buf->length + len > buf->capacity) {
        return -1; // Buffer full
    }
    
    size_t write_pos = (buf->read_pos + buf->length) % buf->capacity;
    
    if (write_pos + len <= buf->capacity) {
        // Simple case: no wraparound
        memcpy(buf->data + write_pos, data, len);
    } else {
        // Need to wrap around
        size_t first_part = buf->capacity - write_pos;
        memcpy(buf->data + write_pos, data, first_part);
        memcpy(buf->data, (char*)data + first_part, len - first_part);
    }
    
    buf->length += len;
    return 0;
}

int buffer_read(circular_buffer_t* buf, void* data, size_t len) {
    if (len > buf->length) {
        len = buf->length; // Read what's available
    }
    
    if (buf->read_pos + len <= buf->capacity) {
        // Simple case: no wraparound
        memcpy(data, buf->data + buf->read_pos, len);
    } else {
        // Need to wrap around
        size_t first_part = buf->capacity - buf->read_pos;
        memcpy(data, buf->data + buf->read_pos, first_part);
        memcpy((char*)data + first_part, buf->data, len - first_part);
    }
    
    buf->read_pos = (buf->read_pos + len) % buf->capacity;
    buf->length -= len;
    return len;
}
```

**Best Practices for I/O Multiplexing:**

1. **Always Check Return Values**
```c
int result = select(max_fd + 1, &readfds, &writefds, NULL, &timeout);
if (result < 0) {
    if (errno == EINTR) {
        continue; // Interrupted by signal, retry
    } else {
        perror("select failed");
        break;
    }
}
```

2. **Handle Partial I/O Operations**
```c
// WRONG: Assumes all data is sent at once
send(sockfd, buffer, 1000, 0);

// CORRECT: Handle partial sends
size_t total_sent = 0;
while (total_sent < data_len) {
    ssize_t sent = send(sockfd, data + total_sent, data_len - total_sent, 0);
    if (sent < 0) {
        if (errno == EAGAIN) {
            // Wait for socket to become writable again
            break;
        } else {
            // Real error
            return -1;
        }
    }
    total_sent += sent;
}
```

3. **Set Appropriate Timeouts**
```c
// For interactive applications
struct timeval timeout;
timeout.tv_sec = 0;
timeout.tv_usec = 100000; // 100ms

// For batch processing
timeout.tv_sec = 1;
timeout.tv_usec = 0; // 1 second
```

4. **Monitor Both Read and Write Events When Needed**
```c
// Initially monitor for read
FD_SET(client_fd, &read_fds);

// When we have data to send, also monitor for write
if (client->send_buffer_len > 0) {
    FD_SET(client_fd, &write_fds);
}

// After sending all data, stop monitoring write
if (client->send_buffer_len == 0) {
    FD_CLR(client_fd, &write_fds);
}
```

### 4. Event-driven I/O

Event-driven I/O represents the pinnacle of scalable network programming. Instead of polling file descriptors, the kernel notifies your application when events occur. This approach scales to tens of thousands of concurrent connections with minimal CPU overhead.

#### Understanding Event Notification

**Traditional Polling Problem:**
```
Application asks: "Is socket ready?"
Kernel responds: "No"
Application asks: "Is socket ready?" (1ms later)
Kernel responds: "No"
Application asks: "Is socket ready?" (1ms later)
Kernel responds: "Yes!"

Problem: Wasted CPU cycles constantly asking
```

**Event-Driven Solution:**
```
Application says: "Tell me when socket is ready"
Kernel responds: "OK, I'll notify you"
... Application does other work ...
Kernel notifies: "Socket X is ready for reading!"
Application processes: Handle socket X

Benefits: CPU only used when work is available
```

#### epoll (Linux) - Edge-triggered and Level-triggered

**epoll Concepts:**

1. **epoll instance**: A kernel data structure that monitors multiple file descriptors
2. **Interest list**: File descriptors and events you want to monitor
3. **Ready list**: File descriptors that have events ready

**Level-triggered vs Edge-triggered:**

```c
// Level-triggered (default): Notified as long as condition is true
// - Socket has data → get notification
// - Read some data, socket still has data → get notification again
// - Read all data → no more notifications

// Edge-triggered: Notified only when condition changes
// - Socket has no data → no notification
// - Data arrives → get notification ONCE
// - More data arrives → get notification ONCE
// - No more notifications until ALL data is read AND new data arrives
```

**Comprehensive epoll Server Example:**
```c
#include <sys/epoll.h>
#include <fcntl.h>

#define MAX_EVENTS 1024
#define BUFFER_SIZE 4096

typedef struct {
    int fd;
    char* recv_buffer;
    size_t recv_capacity;
    size_t recv_length;
    char* send_buffer;
    size_t send_capacity;
    size_t send_length;
    size_t send_offset;
    time_t last_activity;
} epoll_client_t;

typedef struct {
    epoll_client_t* clients;
    size_t capacity;
    size_t count;
} client_pool_t;

client_pool_t* create_client_pool(size_t capacity) {
    client_pool_t* pool = malloc(sizeof(client_pool_t));
    pool->clients = calloc(capacity, sizeof(epoll_client_t));
    pool->capacity = capacity;
    pool->count = 0;
    return pool;
}

epoll_client_t* add_client(client_pool_t* pool, int fd) {
    if (pool->count >= pool->capacity) {
        return NULL;
    }
    
    epoll_client_t* client = &pool->clients[pool->count++];
    client->fd = fd;
    client->recv_buffer = malloc(BUFFER_SIZE);
    client->recv_capacity = BUFFER_SIZE;
    client->recv_length = 0;
    client->send_buffer = malloc(BUFFER_SIZE);
    client->send_capacity = BUFFER_SIZE;
    client->send_length = 0;
    client->send_offset = 0;
    client->last_activity = time(NULL);
    
    return client;
}

void remove_client(client_pool_t* pool, epoll_client_t* client) {
    free(client->recv_buffer);
    free(client->send_buffer);
    
    // Move last client to this position
    if (client != &pool->clients[pool->count - 1]) {
        *client = pool->clients[pool->count - 1];
    }
    pool->count--;
}

epoll_client_t* find_client(client_pool_t* pool, int fd) {
    for (size_t i = 0; i < pool->count; i++) {
        if (pool->clients[i].fd == fd) {
            return &pool->clients[i];
        }
    }
    return NULL;
}

int set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

void handle_client_read(int epoll_fd, epoll_client_t* client) {
    while (1) {
        // Ensure buffer has space
        if (client->recv_length >= client->recv_capacity) {
            // Resize buffer
            client->recv_capacity *= 2;
            client->recv_buffer = realloc(client->recv_buffer, client->recv_capacity);
        }
        
        ssize_t bytes_read = recv(client->fd, 
                                client->recv_buffer + client->recv_length,
                                client->recv_capacity - client->recv_length,
                                0);
        
        if (bytes_read > 0) {
            client->recv_length += bytes_read;
            client->last_activity = time(NULL);
            
            // Process complete messages (assuming line-based protocol)
            char* line_start = client->recv_buffer;
            char* line_end;
            
            while ((line_end = memchr(line_start, '\n', 
                                    client->recv_length - (line_start - client->recv_buffer)))) {
                size_t line_length = line_end - line_start + 1;
                
                // Prepare echo response
                if (client->send_length + line_length <= client->send_capacity) {
                    memcpy(client->send_buffer + client->send_length, line_start, line_length);
                    client->send_length += line_length;
                    
                    // Enable EPOLLOUT to send response
                    struct epoll_event ev;
                    ev.events = EPOLLIN | EPOLLOUT | EPOLLET;
                    ev.data.fd = client->fd;
                    epoll_ctl(epoll_fd, EPOLL_CTL_MOD, client->fd, &ev);
                }
                
                line_start = line_end + 1;
            }
            
            // Remove processed data
            if (line_start > client->recv_buffer) {
                size_t remaining = client->recv_length - (line_start - client->recv_buffer);
                memmove(client->recv_buffer, line_start, remaining);
                client->recv_length = remaining;
            }
            
        } else if (bytes_read == 0) {
            // Connection closed
            printf("Client disconnected\n");
            epoll_ctl(epoll_fd, EPOLL_CTL_DEL, client->fd, NULL);
            close(client->fd);
            return; // Client will be removed by caller
            
        } else {
            // bytes_read < 0
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // No more data available (expected with edge-triggered)
                break;
            } else if (errno == EINTR) {
                // Interrupted, continue
                continue;
            } else {
                // Real error
                perror("recv error");
                epoll_ctl(epoll_fd, EPOLL_CTL_DEL, client->fd, NULL);
                close(client->fd);
                return;
            }
        }
    }
}

void handle_client_write(int epoll_fd, epoll_client_t* client) {
    while (client->send_offset < client->send_length) {
        ssize_t bytes_sent = send(client->fd,
                                client->send_buffer + client->send_offset,
                                client->send_length - client->send_offset,
                                0);
        
        if (bytes_sent > 0) {
            client->send_offset += bytes_sent;
            
        } else if (bytes_sent == 0) {
            // This shouldn't happen with send()
            break;
            
        } else {
            // bytes_sent < 0
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Can't send more right now
                break;
            } else if (errno == EINTR) {
                // Interrupted, continue
                continue;
            } else {
                // Real error
                perror("send error");
                epoll_ctl(epoll_fd, EPOLL_CTL_DEL, client->fd, NULL);
                close(client->fd);
                return;
            }
        }
    }
    
    // Check if all data sent
    if (client->send_offset >= client->send_length) {
        // Reset send buffer
        client->send_length = 0;
        client->send_offset = 0;
        
        // Disable EPOLLOUT (only monitor for reads)
        struct epoll_event ev;
        ev.events = EPOLLIN | EPOLLET;
        ev.data.fd = client->fd;
        epoll_ctl(epoll_fd, EPOLL_CTL_MOD, client->fd, &ev);
    }
}

void epoll_server(int server_fd) {
    int epoll_fd = epoll_create1(EPOLL_CLOEXEC);
    if (epoll_fd < 0) {
        perror("epoll_create1");
        return;
    }
    
    // Add server socket to epoll
    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.fd = server_fd;
    
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, server_fd, &ev) < 0) {
        perror("epoll_ctl: server");
        close(epoll_fd);
        return;
    }
    
    client_pool_t* pool = create_client_pool(10000);
    struct epoll_event events[MAX_EVENTS];
    
    printf("Epoll server listening (edge-triggered)...\n");
    
    while (1) {
        int nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, 1000); // 1 second timeout
        
        if (nfds < 0) {
            if (errno == EINTR) {
                continue;
            }
            perror("epoll_wait");
            break;
        } else if (nfds == 0) {
            // Timeout - do maintenance
            printf("Server heartbeat: %zu clients connected\n", pool->count);
            
            // Check for idle clients
            time_t now = time(NULL);
            for (size_t i = 0; i < pool->count; i++) {
                if (now - pool->clients[i].last_activity > 300) { // 5 minutes
                    printf("Closing idle client\n");
                    epoll_ctl(epoll_fd, EPOLL_CTL_DEL, pool->clients[i].fd, NULL);
                    close(pool->clients[i].fd);
                    remove_client(pool, &pool->clients[i]);
                    i--; // Adjust index after removal
                }
            }
            continue;
        }
        
        for (int i = 0; i < nfds; i++) {
            int fd = events[i].data.fd;
            
            if (fd == server_fd) {
                // New connection(s)
                while (1) {
                    struct sockaddr_in client_addr;
                    socklen_t addr_len = sizeof(client_addr);
                    
                    int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addr_len);
                    if (client_fd < 0) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) {
                            // No more connections
                            break;
                        } else {
                            perror("accept");
                            break;
                        }
                    }
                    
                    // Set non-blocking
                    if (set_nonblocking(client_fd) < 0) {
                        close(client_fd);
                        continue;
                    }
                    
                    // Add client to pool
                    epoll_client_t* client = add_client(pool, client_fd);
                    if (!client) {
                        printf("Client pool full\n");
                        close(client_fd);
                        continue;
                    }
                    
                    // Add to epoll (edge-triggered)
                    ev.events = EPOLLIN | EPOLLET;
                    ev.data.fd = client_fd;
                    
                    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &ev) < 0) {
                        perror("epoll_ctl: client");
                        close(client_fd);
                        remove_client(pool, client);
                        continue;
                    }
                    
                    printf("New client connected: %s:%d (total: %zu)\n",
                           inet_ntoa(client_addr.sin_addr),
                           ntohs(client_addr.sin_port),
                           pool->count);
                }
                
            } else {
                // Client socket event
                epoll_client_t* client = find_client(pool, fd);
                if (!client) {
                    // Client not found, remove from epoll
                    epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, NULL);
                    close(fd);
                    continue;
                }
                
                if (events[i].events & (EPOLLERR | EPOLLHUP)) {
                    // Error or hangup
                    printf("Client error/hangup\n");
                    epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, NULL);
                    close(fd);
                    remove_client(pool, client);
                    continue;
                }
                
                if (events[i].events & EPOLLIN) {
                    handle_client_read(epoll_fd, client);
                    
                    // Check if client was removed
                    if (find_client(pool, fd) == NULL) {
                        continue;
                    }
                }
                
                if (events[i].events & EPOLLOUT) {
                    handle_client_write(epoll_fd, client);
                }
            }
        }
    }
    
    // Cleanup
    for (size_t i = 0; i < pool->count; i++) {
        close(pool->clients[i].fd);
    }
    free(pool->clients);
    free(pool);
    close(epoll_fd);
}
```

#### kqueue (BSD/macOS)

kqueue provides similar functionality to epoll but with a different API. It's available on FreeBSD, OpenBSD, NetBSD, and macOS.

**kqueue Key Concepts:**
- **kqueue**: The kernel event queue
- **kevent**: Individual events to monitor or events that occurred
- **Filters**: Types of events (EVFILT_READ, EVFILT_WRITE, etc.)

**Comprehensive kqueue Server Example:**
```c
#include <sys/types.h>
#include <sys/event.h>
#include <sys/time.h>

#define MAX_EVENTS 1024

typedef struct {
    int fd;
    char buffer[4096];
    size_t buffer_len;
    time_t last_activity;
} kqueue_client_t;

void kqueue_server(int server_fd) {
    int kq = kqueue();
    if (kq < 0) {
        perror("kqueue");
        return;
    }
    
    // Add server socket to kqueue
    struct kevent change;
    EV_SET(&change, server_fd, EVFILT_READ, EV_ADD, 0, 0, NULL);
    
    if (kevent(kq, &change, 1, NULL, 0, NULL) < 0) {
        perror("kevent: server socket");
        close(kq);
        return;
    }
    
    // Client management
    kqueue_client_t* clients = calloc(10000, sizeof(kqueue_client_t));
    size_t max_clients = 10000;
    size_t client_count = 0;
    
    struct kevent events[MAX_EVENTS];
    struct timespec timeout;
    timeout.tv_sec = 1;
    timeout.tv_nsec = 0;
    
    printf("Kqueue server listening...\n");
    
    while (1) {
        int nev = kevent(kq, NULL, 0, events, MAX_EVENTS, &timeout);
        
        if (nev < 0) {
            if (errno == EINTR) {
                continue;
            }
            perror("kevent");
            break;
        } else if (nev == 0) {
            // Timeout
            printf("Server heartbeat: %zu clients\n", client_count);
            continue;
        }
        
        for (int i = 0; i < nev; i++) {
            int fd = events[i].ident;
            
            if (fd == server_fd) {
                // New connection
                struct sockaddr_in client_addr;
                socklen_t addr_len = sizeof(client_addr);
                
                int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addr_len);
                if (client_fd >= 0) {
                    if (client_count < max_clients) {
                        // Add client to array
                        size_t slot = client_count++;
                        clients[slot].fd = client_fd;
                        clients[slot].buffer_len = 0;
                        clients[slot].last_activity = time(NULL);
                        
                        // Add to kqueue for reading
                        EV_SET(&change, client_fd, EVFILT_READ, EV_ADD, 0, 0, &clients[slot]);
                        
                        if (kevent(kq, &change, 1, NULL, 0, NULL) < 0) {
                            perror("kevent: client add");
                            close(client_fd);
                            client_count--;
                        } else {
                            printf("New client connected: %s:%d (total: %zu)\n",
                                   inet_ntoa(client_addr.sin_addr),
                                   ntohs(client_addr.sin_port),
                                   client_count);
                        }
                    } else {
                        printf("Max clients reached\n");
                        close(client_fd);
                    }
                }
                
            } else {
                // Client event
                kqueue_client_t* client = (kqueue_client_t*)events[i].udata;
                
                if (events[i].filter == EVFILT_READ) {
                    if (events[i].flags & EV_EOF) {
                        // Connection closed
                        printf("Client disconnected\n");
                        close(client->fd);
                        
                        // Remove from array (move last client here)
                        if (client != &clients[client_count - 1]) {
                            *client = clients[client_count - 1];
                            
                            // Update kqueue udata pointer
                            EV_SET(&change, client->fd, EVFILT_READ, EV_ADD, 0, 0, client);
                            kevent(kq, &change, 1, NULL, 0, NULL);
                        }
                        client_count--;
                        
                    } else {
                        // Data available
                        ssize_t bytes_read = recv(client->fd, client->buffer, 
                                                sizeof(client->buffer), 0);
                        
                        if (bytes_read > 0) {
                            // Echo back
                            send(client->fd, client->buffer, bytes_read, 0);
                            client->last_activity = time(NULL);
                            
                        } else if (bytes_read == 0) {
                            // Connection closed
                            printf("Client disconnected\n");
                            close(client->fd);
                            
                            // Remove from array
                            if (client != &clients[client_count - 1]) {
                                *client = clients[client_count - 1];
                                
                                // Update kqueue udata pointer
                                EV_SET(&change, client->fd, EVFILT_READ, EV_ADD, 0, 0, client);
                                kevent(kq, &change, 1, NULL, 0, NULL);
                            }
                            client_count--;
                            
                        } else {
                            // Error
                            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                                perror("recv");
                                close(client->fd);
                                
                                // Remove from array
                                if (client != &clients[client_count - 1]) {
                                    *client = clients[client_count - 1];
                                    
                                    // Update kqueue udata pointer  
                                    EV_SET(&change, client->fd, EVFILT_READ, EV_ADD, 0, 0, client);
                                    kevent(kq, &change, 1, NULL, 0, NULL);
                                }
                                client_count--;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Cleanup
    for (size_t i = 0; i < client_count; i++) {
        close(clients[i].fd);
    }
    free(clients);
    close(kq);
}
```

#### IOCP (Windows)

I/O Completion Ports (IOCP) is Windows' high-performance I/O model. It's based on completion notifications rather than readiness notifications.

**IOCP Key Concepts:**
- **Completion Port**: Kernel object that queues completed I/O operations
- **Overlapped I/O**: Asynchronous I/O operations
- **Completion Packets**: Notifications of completed operations

**Basic IOCP Server Structure:**
```c
#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>

#define BUFFER_SIZE 4096

typedef struct {
    OVERLAPPED overlapped;
    SOCKET socket;
    char buffer[BUFFER_SIZE];
    DWORD bytes_transferred;
    DWORD flags;
    enum { OP_READ, OP_WRITE } operation;
} per_io_data_t;

typedef struct {
    SOCKET socket;
    struct sockaddr_in addr;
} per_handle_data_t;

void iocp_server(SOCKET server_socket) {
    // Create I/O completion port
    HANDLE iocp = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, 0);
    if (iocp == NULL) {
        printf("CreateIoCompletionPort failed: %d\n", GetLastError());
        return;
    }
    
    // Associate server socket with completion port
    if (CreateIoCompletionPort((HANDLE)server_socket, iocp, (ULONG_PTR)NULL, 0) == NULL) {
        printf("CreateIoCompletionPort (server) failed: %d\n", GetLastError());
        CloseHandle(iocp);
        return;
    }
    
    printf("IOCP server listening...\n");
    
    while (1) {
        DWORD bytes_transferred;
        ULONG_PTR completion_key;
        OVERLAPPED* overlapped;
        
        // Wait for completion
        BOOL result = GetQueuedCompletionStatus(iocp, &bytes_transferred, 
                                              &completion_key, &overlapped, 
                                              1000); // 1 second timeout
        
        if (!result) {
            DWORD error = GetLastError();
            if (error == WAIT_TIMEOUT) {
                printf("Server heartbeat\n");
                continue;
            } else {
                printf("GetQueuedCompletionStatus failed: %d\n", error);
                break;
            }
        }
        
        if (overlapped == NULL) {
            // Server socket notification (new connection)
            struct sockaddr_in client_addr;
            int addr_len = sizeof(client_addr);
            
            SOCKET client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &addr_len);
            if (client_socket != INVALID_SOCKET) {
                // Create per-handle data
                per_handle_data_t* handle_data = malloc(sizeof(per_handle_data_t));
                handle_data->socket = client_socket;
                handle_data->addr = client_addr;
                
                // Associate client socket with completion port
                if (CreateIoCompletionPort((HANDLE)client_socket, iocp, 
                                         (ULONG_PTR)handle_data, 0) == NULL) {
                    printf("CreateIoCompletionPort (client) failed: %d\n", GetLastError());
                    closesocket(client_socket);
                    free(handle_data);
                    continue;
                }
                
                // Start async read
                per_io_data_t* io_data = malloc(sizeof(per_io_data_t));
                memset(&io_data->overlapped, 0, sizeof(OVERLAPPED));
                io_data->socket = client_socket;
                io_data->operation = OP_READ;
                io_data->flags = 0;
                
                WSABUF wsa_buf;
                wsa_buf.len = BUFFER_SIZE;
                wsa_buf.buf = io_data->buffer;
                
                if (WSARecv(client_socket, &wsa_buf, 1, &io_data->bytes_transferred,
                           &io_data->flags, &io_data->overlapped, NULL) == SOCKET_ERROR) {
                    if (WSAGetLastError() != WSA_IO_PENDING) {
                        printf("WSARecv failed: %d\n", WSAGetLastError());
                        closesocket(client_socket);
                        free(handle_data);
                        free(io_data);
                    }
                }
                
                printf("New client connected\n");
            }
            
        } else {
            // I/O operation completed
            per_io_data_t* io_data = CONTAINING_RECORD(overlapped, per_io_data_t, overlapped);
            per_handle_data_t* handle_data = (per_handle_data_t*)completion_key;
            
            if (bytes_transferred == 0) {
                // Connection closed
                printf("Client disconnected\n");
                closesocket(handle_data->socket);
                free(handle_data);
                free(io_data);
                continue;
            }
            
            if (io_data->operation == OP_READ) {
                // Data received, echo it back
                io_data->operation = OP_WRITE;
                io_data->bytes_transferred = bytes_transferred;
                
                WSABUF wsa_buf;
                wsa_buf.len = bytes_transferred;
                wsa_buf.buf = io_data->buffer;
                
                if (WSASend(io_data->socket, &wsa_buf, 1, NULL, 0, 
                           &io_data->overlapped, NULL) == SOCKET_ERROR) {
                    if (WSAGetLastError() != WSA_IO_PENDING) {
                        printf("WSASend failed: %d\n", WSAGetLastError());
                        closesocket(handle_data->socket);
                        free(handle_data);
                        free(io_data);
                    }
                }
                
            } else if (io_data->operation == OP_WRITE) {
                // Send completed, start another read
                io_data->operation = OP_READ;
                memset(&io_data->overlapped, 0, sizeof(OVERLAPPED));
                
                WSABUF wsa_buf;
                wsa_buf.len = BUFFER_SIZE;
                wsa_buf.buf = io_data->buffer;
                io_data->flags = 0;
                
                if (WSARecv(io_data->socket, &wsa_buf, 1, &io_data->bytes_transferred,
                           &io_data->flags, &io_data->overlapped, NULL) == SOCKET_ERROR) {
                    if (WSAGetLastError() != WSA_IO_PENDING) {
                        printf("WSARecv failed: %d\n", WSAGetLastError());
                        closesocket(handle_data->socket);
                        free(handle_data);
                        free(io_data);
                    }
                }
            }
        }
    }
    
    CloseHandle(iocp);
}
#endif // _WIN32
```

#### Event Notification Mechanisms

**Summary of Event Notification Models:**

| Model | Platform | Scalability | Complexity | Edge/Level |
|-------|----------|-------------|------------|------------|
| **select()** | POSIX | Low (1024 FD limit) | Medium | Level |
| **poll()** | POSIX | Medium (no FD limit) | Medium | Level |
| **epoll** | Linux | High (100K+ connections) | High | Both |
| **kqueue** | BSD/macOS | High (100K+ connections) | High | Both |
| **IOCP** | Windows | High (100K+ connections) | High | Completion |

**Performance Characteristics:**

```c
// Benchmark results (approximate, varies by system):
// 
// Concurrent Connections: 1,000
// - select(): ~50% CPU usage, frequent system calls
// - poll(): ~40% CPU usage, better than select
// - epoll(): ~5% CPU usage, excellent efficiency
//
// Concurrent Connections: 10,000  
// - select(): Usually impossible (FD_SETSIZE limit)
// - poll(): ~80% CPU usage, performance degrades
// - epoll(): ~15% CPU usage, scales well
//
// Concurrent Connections: 100,000
// - select(): Impossible
// - poll(): Impossible or extremely slow
// - epoll(): ~30% CPU usage, still manageable
```

**Choosing the Right Event Model:**

```c
#ifdef __linux__
    // Use epoll for maximum performance on Linux
    epoll_server(server_fd);
#elif defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__) || defined(__APPLE__)
    // Use kqueue on BSD systems and macOS
    kqueue_server(server_fd);
#elif defined(_WIN32)
    // Use IOCP on Windows
    iocp_server(server_socket);
#else
    // Fall back to poll() for portability
    poll_based_server(server_fd);
#endif
```

## Practical Exercises

### Exercise 1: Blocking vs Non-blocking Comparison

**Objective:** Compare performance and behavior of blocking vs non-blocking I/O models.

**Implementation Tasks:**
```c
// Task 1: Implement blocking echo server
void blocking_echo_server(int port) {
    // TODO: Create server socket
    // TODO: Accept connections in loop
    // TODO: Handle each client in separate thread
    // TODO: Echo received data back to client
}

// Task 2: Implement non-blocking echo server  
void nonblocking_echo_server(int port) {
    // TODO: Create server socket and set non-blocking
    // TODO: Accept connections without blocking
    // TODO: Handle all clients in single thread
    // TODO: Use proper error handling for EAGAIN/EWOULDBLOCK
}

// Task 3: Performance measurement
typedef struct {
    double connection_time;
    double request_response_time;
    int successful_connections;
    int failed_connections;
} performance_metrics_t;

performance_metrics_t benchmark_server(const char* server_type, int port, int num_clients);
```

**Test Scenarios:**
1. **Low Load Test:** 10 concurrent clients, 100 requests each
2. **Medium Load Test:** 100 concurrent clients, 50 requests each  
3. **High Load Test:** 1000 concurrent clients, 10 requests each
4. **Burst Test:** Create 500 connections simultaneously
5. **Sustained Load:** 50 clients continuously for 5 minutes

**Expected Results:**
- Blocking server: Good performance up to ~100 clients, then degrades
- Non-blocking server: Better resource usage, handles more clients
- Memory usage comparison: Blocking uses more memory (thread stacks)

### Exercise 2: Select-based Multi-Protocol Server

**Objective:** Build a server that handles multiple protocols simultaneously using select().

**Requirements:**
```c
// Server should handle:
// 1. HTTP requests on port 8080
// 2. Echo protocol on port 8081  
// 3. Chat protocol on port 8082
// 4. Admin interface on port 8083

typedef struct {
    int fd;
    enum { PROTO_HTTP, PROTO_ECHO, PROTO_CHAT, PROTO_ADMIN } protocol;
    char buffer[4096];
    size_t buffer_len;
    // Protocol-specific state
    union {
        struct {
            char method[16];
            char path[256]; 
            int content_length;
            int headers_complete;
        } http;
        struct {
            char nickname[32];
            int authenticated;
        } chat;
    } state;
} multi_client_t;

void multi_protocol_server();
```

**Implementation Steps:**
1. Create multiple listening sockets for different protocols
2. Use select() to monitor all sockets simultaneously
3. Implement protocol detection and handling
4. Add proper connection management and cleanup
5. Include statistics and monitoring

**Advanced Features:**
- Protocol auto-detection based on first received data
- Cross-protocol communication (chat users can query HTTP stats)
- Rate limiting per protocol
- Connection pooling and reuse

### Exercise 3: Epoll-based High-Performance Server

**Objective:** Create a production-ready server using epoll that can handle 10,000+ concurrent connections.

**Architecture:**
```c
// Multi-threaded epoll server design:
// - Main thread: Accepts new connections
// - Worker threads: Handle client I/O using epoll
// - Load balancing: Distribute clients across workers

typedef struct {
    int epoll_fd;
    pthread_t thread_id;
    client_pool_t* clients;
    atomic_int active_connections;
    atomic_long bytes_received;
    atomic_long bytes_sent;
} worker_thread_t;

typedef struct {
    worker_thread_t* workers;
    int num_workers;
    int server_fd;
    atomic_int total_connections;
    atomic_int next_worker; // Round-robin assignment
} server_context_t;

void* worker_thread(void* arg);
void* acceptor_thread(void* arg);
int create_high_performance_server(int port, int num_workers);
```

**Performance Requirements:**
- Handle 10,000 concurrent connections
- Process 100,000 messages per second
- Memory usage < 100 MB for 10K connections
- Average latency < 1ms
- 99th percentile latency < 10ms

**Testing Tools:**
```bash
# Use these tools to test your server:

# 1. Connection flood test
./connection_flood_test localhost 8080 10000

# 2. Sustained load test  
./load_test --host localhost --port 8080 --connections 5000 --duration 300s

# 3. Latency measurement
./latency_test --host localhost --port 8080 --connections 1000 --requests 100000

# 4. Memory profiling
valgrind --tool=massif ./your_server
```

### Exercise 4: Cross-platform I/O Abstraction Layer

**Objective:** Create a unified interface that works across Linux (epoll), macOS (kqueue), and Windows (IOCP).

**Interface Design:**
```c
// Abstract I/O event system
typedef struct io_context io_context_t;
typedef struct io_event io_event_t;

// Event types
#define IO_EVENT_READ    (1 << 0)
#define IO_EVENT_WRITE   (1 << 1)  
#define IO_EVENT_ERROR   (1 << 2)
#define IO_EVENT_HANGUP  (1 << 3)

// Callback function type
typedef void (*io_callback_t)(io_context_t* ctx, int fd, int events, void* user_data);

// Public API
io_context_t* io_context_create(void);
void io_context_destroy(io_context_t* ctx);

int io_add_fd(io_context_t* ctx, int fd, int events, io_callback_t callback, void* user_data);
int io_modify_fd(io_context_t* ctx, int fd, int events);
int io_remove_fd(io_context_t* ctx, int fd);

int io_run(io_context_t* ctx, int timeout_ms);
void io_stop(io_context_t* ctx);

// Platform-specific implementations
#ifdef __linux__
    // epoll implementation
#elif defined(__APPLE__) || defined(__FreeBSD__)
    // kqueue implementation  
#elif defined(_WIN32)
    // IOCP implementation
#else
    // poll() fallback implementation
#endif
```

**Implementation Requirements:**
1. Same API works on all platforms
2. Performance equivalent to native APIs
3. Handle edge cases consistently
4. Comprehensive error reporting
5. Thread-safe operations

**Testing Strategy:**
```c
// Cross-platform test suite
void test_basic_operations(void);
void test_edge_triggered_behavior(void);
void test_high_load_scenarios(void);
void test_error_conditions(void);
void benchmark_performance(void);

// Platform-specific tests
void test_epoll_specific_features(void);   // Linux only
void test_kqueue_specific_features(void);  // BSD/macOS only
void test_iocp_specific_features(void);    // Windows only
```

### Exercise 5: Real-time Chat Server

**Objective:** Build a complete chat application demonstrating practical use of I/O models.

**Features:**
- Multiple chat rooms
- User authentication
- Message history
- Private messaging
- File sharing
- Real-time notifications

**Protocol Design:**
```c
// JSON-based message protocol
typedef struct {
    char type[16];        // "join", "message", "leave", "private", etc.
    char user[32];        // Username
    char room[32];        // Room name (if applicable)
    char data[1024];      // Message content
    time_t timestamp;     // Message timestamp
} chat_message_t;

// Server state management
typedef struct {
    char name[32];
    hashtable_t* users;   // username -> user_info_t*
    message_history_t* history;
} chat_room_t;

typedef struct {
    int fd;
    char username[32];
    char current_room[32];
    time_t last_activity;
    user_permissions_t permissions;
} user_info_t;
```

**Advanced Features:**
```c
// Real-time features using I/O models
void broadcast_to_room(const char* room, const chat_message_t* msg);
void send_private_message(const char* from, const char* to, const char* msg);
void notify_user_joined(const char* room, const char* username);
void handle_file_upload(int client_fd, const char* filename, size_t filesize);

// Performance monitoring
typedef struct {
    atomic_long messages_sent;
    atomic_long messages_received;
    atomic_long bytes_transferred;
    atomic_int active_users;
    atomic_int active_rooms;
    double avg_response_time;
} chat_server_stats_t;
```

### Exercise 6: HTTP Load Balancer

**Objective:** Implement a high-performance HTTP load balancer using event-driven I/O.

**Core Features:**
- Multiple backend servers
- Health checking
- Load balancing algorithms (round-robin, least connections, weighted)
- Connection pooling
- SSL/TLS termination

**Implementation Structure:**
```c
typedef struct {
    char host[256];
    int port;
    int weight;
    atomic_int active_connections;
    atomic_int total_requests;
    double avg_response_time;
    time_t last_health_check;
    int healthy;
} backend_server_t;

typedef struct {
    int client_fd;
    int backend_fd;
    backend_server_t* backend;
    char* client_buffer;
    size_t client_buffer_len;
    char* backend_buffer;
    size_t backend_buffer_len;
    time_t start_time;
} proxy_connection_t;

// Load balancing algorithms
backend_server_t* round_robin_select(backend_server_t* servers, int count);
backend_server_t* least_connections_select(backend_server_t* servers, int count);
backend_server_t* weighted_select(backend_server_t* servers, int count);
```

**Testing Scenarios:**
1. **Basic Functionality:** Single backend, verify request forwarding
2. **Load Distribution:** Multiple backends, verify load balancing
3. **Failover:** Kill backend server, verify failover behavior
4. **Performance:** Handle 50,000 concurrent connections
5. **Health Checking:** Backends going up/down dynamically

## Code Examples

### 1. Blocking I/O with Timeout

```c
#include <sys/socket.h>
#include <sys/time.h>

int recv_with_timeout(int sockfd, void* buffer, size_t len, int timeout_sec) {
    fd_set readfds;
    struct timeval timeout;
    
    FD_ZERO(&readfds);
    FD_SET(sockfd, &readfds);
    
    timeout.tv_sec = timeout_sec;
    timeout.tv_usec = 0;
    
    int result = select(sockfd + 1, &readfds, NULL, NULL, &timeout);
    if (result < 0) {
        perror("Select failed");
        return -1;
    } else if (result == 0) {
        // Timeout
        errno = ETIMEDOUT;
        return -1;
    }
    
    return recv(sockfd, buffer, len, 0);
}

int send_with_timeout(int sockfd, const void* buffer, size_t len, int timeout_sec) {
    fd_set writefds;
    struct timeval timeout;
    
    FD_ZERO(&writefds);
    FD_SET(sockfd, &writefds);
    
    timeout.tv_sec = timeout_sec;
    timeout.tv_usec = 0;
    
    int result = select(sockfd + 1, NULL, &writefds, NULL, &timeout);
    if (result < 0) {
        perror("Select failed");
        return -1;
    } else if (result == 0) {
        // Timeout
        errno = ETIMEDOUT;
        return -1;
    }
    
    return send(sockfd, buffer, len, 0);
}
```

### 2. Non-blocking I/O

```c
#include <fcntl.h>
#include <errno.h>

int set_nonblocking(int sockfd) {
    int flags = fcntl(sockfd, F_GETFL, 0);
    if (flags < 0) {
        perror("fcntl F_GETFL");
        return -1;
    }
    
    if (fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) < 0) {
        perror("fcntl F_SETFL O_NONBLOCK");
        return -1;
    }
    
    return 0;
}

ssize_t nonblocking_send_all(int sockfd, const void* buffer, size_t length) {
    const char* ptr = (const char*)buffer;
    size_t total_sent = 0;
    
    while (total_sent < length) {
        ssize_t sent = send(sockfd, ptr + total_sent, length - total_sent, 0);
        
        if (sent < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Socket buffer full, need to wait
                fd_set writefds;
                FD_ZERO(&writefds);
                FD_SET(sockfd, &writefds);
                
                if (select(sockfd + 1, NULL, &writefds, NULL, NULL) < 0) {
                    return -1;
                }
                continue;
            } else if (errno == EINTR) {
                continue;  // Interrupted, retry
            } else {
                return -1;  // Real error
            }
        }
        
        if (sent == 0) break;  // Connection closed
        total_sent += sent;
    }
    
    return total_sent;
}

ssize_t nonblocking_recv_available(int sockfd, void* buffer, size_t length) {
    ssize_t received = recv(sockfd, buffer, length, 0);
    
    if (received < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // No data available right now
            return 0;
        } else if (errno == EINTR) {
            // Interrupted, try again
            return nonblocking_recv_available(sockfd, buffer, length);
        } else {
            // Real error
            return -1;
        }
    }
    
    return received;
}
```

### 3. Select-based I/O Multiplexing

```c
#include <sys/select.h>

#define MAX_CLIENTS 100

typedef struct {
    int sockfd;
    char buffer[1024];
    size_t buffer_len;
    struct sockaddr_in addr;
} client_t;

void select_based_server(int server_fd) {
    client_t clients[MAX_CLIENTS];
    fd_set master_set, read_set, write_set;
    int max_fd = server_fd;
    int client_count = 0;
    
    // Initialize
    FD_ZERO(&master_set);
    FD_SET(server_fd, &master_set);
    
    for (int i = 0; i < MAX_CLIENTS; i++) {
        clients[i].sockfd = -1;
    }
    
    while (1) {
        read_set = master_set;
        FD_ZERO(&write_set);
        
        // Set write set for clients with data to send
        for (int i = 0; i < MAX_CLIENTS; i++) {
            if (clients[i].sockfd != -1 && clients[i].buffer_len > 0) {
                FD_SET(clients[i].sockfd, &write_set);
            }
        }
        
        int activity = select(max_fd + 1, &read_set, &write_set, NULL, NULL);
        if (activity < 0) {
            perror("Select failed");
            break;
        }
        
        // Check for new connections
        if (FD_ISSET(server_fd, &read_set)) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            
            int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
            if (client_fd >= 0) {
                // Find free slot
                int slot = -1;
                for (int i = 0; i < MAX_CLIENTS; i++) {
                    if (clients[i].sockfd == -1) {
                        slot = i;
                        break;
                    }
                }
                
                if (slot != -1) {
                    clients[slot].sockfd = client_fd;
                    clients[slot].buffer_len = 0;
                    clients[slot].addr = client_addr;
                    
                    FD_SET(client_fd, &master_set);
                    if (client_fd > max_fd) {
                        max_fd = client_fd;
                    }
                    
                    printf("New client connected: %s:%d\n",
                           inet_ntoa(client_addr.sin_addr),
                           ntohs(client_addr.sin_port));
                } else {
                    printf("Max clients reached, rejecting connection\n");
                    close(client_fd);
                }
            }
        }
        
        // Check client sockets for data
        for (int i = 0; i < MAX_CLIENTS; i++) {
            if (clients[i].sockfd == -1) continue;
            
            int client_fd = clients[i].sockfd;
            
            // Check for incoming data
            if (FD_ISSET(client_fd, &read_set)) {
                char temp_buffer[1024];
                ssize_t bytes_read = recv(client_fd, temp_buffer, sizeof(temp_buffer), 0);
                
                if (bytes_read <= 0) {
                    // Client disconnected
                    printf("Client disconnected\n");
                    close(client_fd);
                    FD_CLR(client_fd, &master_set);
                    clients[i].sockfd = -1;
                    clients[i].buffer_len = 0;
                } else {
                    // Echo data back (store in buffer for writing)
                    if (bytes_read <= sizeof(clients[i].buffer)) {
                        memcpy(clients[i].buffer, temp_buffer, bytes_read);
                        clients[i].buffer_len = bytes_read;
                    }
                }
            }
            
            // Check for outgoing data
            if (FD_ISSET(client_fd, &write_set) && clients[i].buffer_len > 0) {
                ssize_t bytes_sent = send(client_fd, clients[i].buffer, 
                                        clients[i].buffer_len, 0);
                
                if (bytes_sent > 0) {
                    clients[i].buffer_len = 0;  // Clear buffer after sending
                }
            }
        }
    }
}
```

### 4. Poll-based I/O Multiplexing

```c
#include <poll.h>

#define MAX_CLIENTS 1000

void poll_based_server(int server_fd) {
    struct pollfd fds[MAX_CLIENTS];
    int nfds = 1;
    
    // Initialize with server socket
    fds[0].fd = server_fd;
    fds[0].events = POLLIN;
    
    // Initialize client slots
    for (int i = 1; i < MAX_CLIENTS; i++) {
        fds[i].fd = -1;
    }
    
    while (1) {
        int poll_count = poll(fds, nfds, -1);  // Block indefinitely
        
        if (poll_count < 0) {
            perror("Poll failed");
            break;
        }
        
        // Check server socket for new connections
        if (fds[0].revents & POLLIN) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            
            int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
            if (client_fd >= 0) {
                // Find free slot
                if (nfds < MAX_CLIENTS) {
                    fds[nfds].fd = client_fd;
                    fds[nfds].events = POLLIN;
                    nfds++;
                    
                    printf("New client connected, total clients: %d\n", nfds - 1);
                } else {
                    printf("Max clients reached\n");
                    close(client_fd);
                }
            }
        }
        
        // Check client sockets
        for (int i = 1; i < nfds; i++) {
            if (fds[i].fd == -1) continue;
            
            if (fds[i].revents & POLLIN) {
                char buffer[1024];
                ssize_t bytes_read = recv(fds[i].fd, buffer, sizeof(buffer), 0);
                
                if (bytes_read <= 0) {
                    // Client disconnected
                    printf("Client disconnected\n");
                    close(fds[i].fd);
                    
                    // Remove from array by moving last element here
                    fds[i] = fds[nfds - 1];
                    nfds--;
                    i--;  // Recheck this position
                } else {
                    // Echo back
                    send(fds[i].fd, buffer, bytes_read, 0);
                }
            }
            
            if (fds[i].revents & (POLLERR | POLLHUP | POLLNVAL)) {
                // Error or hangup
                printf("Client error/hangup\n");
                close(fds[i].fd);
                fds[i] = fds[nfds - 1];
                nfds--;
                i--;
            }
        }
    }
}
```

### 5. Epoll-based Event-driven I/O (Linux)

```c
#include <sys/epoll.h>

#define MAX_EVENTS 1000

typedef struct {
    int fd;
    char buffer[1024];
    size_t buffer_pos;
    size_t buffer_len;
} connection_t;

void epoll_based_server(int server_fd) {
    int epoll_fd = epoll_create1(EPOLL_CLOEXEC);
    if (epoll_fd < 0) {
        perror("epoll_create1");
        return;
    }
    
    // Add server socket to epoll
    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.fd = server_fd;
    
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, server_fd, &ev) < 0) {
        perror("epoll_ctl: server_fd");
        close(epoll_fd);
        return;
    }
    
    struct epoll_event events[MAX_EVENTS];
    
    while (1) {
        int nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);
        
        if (nfds < 0) {
            perror("epoll_wait");
            break;
        }
        
        for (int i = 0; i < nfds; i++) {
            int fd = events[i].data.fd;
            
            if (fd == server_fd) {
                // New connection
                struct sockaddr_in client_addr;
                socklen_t client_len = sizeof(client_addr);
                
                int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
                if (client_fd >= 0) {
                    // Set non-blocking
                    set_nonblocking(client_fd);
                    
                    // Add to epoll
                    ev.events = EPOLLIN | EPOLLET;  // Edge-triggered
                    ev.data.fd = client_fd;
                    
                    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &ev) < 0) {
                        perror("epoll_ctl: client_fd");
                        close(client_fd);
                    } else {
                        printf("New client connected\n");
                    }
                }
            } else {
                // Client data or error
                if (events[i].events & EPOLLIN) {
                    // Data available for reading
                    char buffer[1024];
                    
                    while (1) {
                        ssize_t bytes_read = recv(fd, buffer, sizeof(buffer), 0);
                        
                        if (bytes_read < 0) {
                            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                                // No more data available
                                break;
                            } else {
                                perror("recv");
                                goto close_client;
                            }
                        } else if (bytes_read == 0) {
                            // Client closed connection
                            printf("Client disconnected\n");
                            goto close_client;
                        } else {
                            // Echo data back
                            if (send(fd, buffer, bytes_read, 0) < 0) {
                                perror("send");
                                goto close_client;
                            }
                        }
                    }
                    continue;
                    
                close_client:
                    epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, NULL);
                    close(fd);
                }
                
                if (events[i].events & (EPOLLERR | EPOLLHUP)) {
                    // Error or hangup
                    printf("Client error/hangup\n");
                    epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, NULL);
                    close(fd);
                }
            }
        }
    }
    
    close(epoll_fd);
}
```

### 6. Kqueue-based Event-driven I/O (BSD/macOS)

```c
#include <sys/event.h>

void kqueue_based_server(int server_fd) {
    int kq = kqueue();
    if (kq < 0) {
        perror("kqueue");
        return;
    }
    
    // Add server socket to kqueue
    struct kevent change;
    EV_SET(&change, server_fd, EVFILT_READ, EV_ADD, 0, 0, NULL);
    
    if (kevent(kq, &change, 1, NULL, 0, NULL) < 0) {
        perror("kevent: server_fd");
        close(kq);
        return;
    }
    
    struct kevent events[MAX_EVENTS];
    
    while (1) {
        int nev = kevent(kq, NULL, 0, events, MAX_EVENTS, NULL);
        
        if (nev < 0) {
            perror("kevent");
            break;
        }
        
        for (int i = 0; i < nev; i++) {
            int fd = events[i].ident;
            
            if (fd == server_fd) {
                // New connection
                struct sockaddr_in client_addr;
                socklen_t client_len = sizeof(client_addr);
                
                int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
                if (client_fd >= 0) {
                    // Add client to kqueue
                    EV_SET(&change, client_fd, EVFILT_READ, EV_ADD, 0, 0, NULL);
                    
                    if (kevent(kq, &change, 1, NULL, 0, NULL) < 0) {
                        perror("kevent: client_fd");
                        close(client_fd);
                    } else {
                        printf("New client connected\n");
                    }
                }
            } else {
                // Client data
                if (events[i].filter == EVFILT_READ) {
                    char buffer[1024];
                    ssize_t bytes_read = recv(fd, buffer, sizeof(buffer), 0);
                    
                    if (bytes_read <= 0) {
                        // Client disconnected or error
                        printf("Client disconnected\n");
                        close(fd);
                    } else {
                        // Echo back
                        send(fd, buffer, bytes_read, 0);
                    }
                }
            }
        }
    }
    
    close(kq);
}
```

## I/O Model Comparison

### Comprehensive Performance Analysis

| Metric | Blocking I/O | Non-blocking I/O | select() | poll() | epoll/kqueue | IOCP |
|--------|--------------|------------------|----------|---------|--------------|------|
| **Max Connections** | ~1,000 | ~1,000 | ~1,000 | ~10,000 | ~100,000+ | ~100,000+ |
| **CPU Usage (10K conn)** | High | Medium | High | High | Low | Low |
| **Memory per Connection** | ~8MB | ~4KB | ~4KB | ~4KB | ~4KB | ~8KB |
| **Scalability** | Poor | Fair | Poor | Fair | Excellent | Excellent |
| **Complexity** | Low | Medium | Medium | Medium | High | High |
| **Portability** | High | High | High | High | Linux only | Windows only |
| **Learning Curve** | Easy | Medium | Medium | Medium | Hard | Hard |

### Detailed Performance Characteristics

#### Connection Handling Capacity
```c
// Approximate limits based on typical server hardware (16GB RAM, 8 cores):

// Blocking I/O (thread-per-connection)
// - Memory limit: 16GB / 8MB = 2,000 threads
// - OS limit: ~1,000-4,000 threads (varies by OS)
// - Performance degrades significantly after 100-500 concurrent connections

// Non-blocking I/O with polling
// - Memory limit: Much higher (4KB per connection)
// - CPU becomes bottleneck due to polling overhead
// - Practical limit: ~1,000 connections

// select() based
// - Hard limit: FD_SETSIZE (usually 1024)
// - Performance degrades linearly with number of file descriptors
// - Not suitable for high-concurrency applications

// poll() based  
// - No hard limit on file descriptors
// - Performance degrades quadratically O(n) with connections
// - Practical limit: ~5,000-10,000 connections

// epoll/kqueue based
// - Performance scales O(1) with active connections
// - Memory usage: ~4KB per connection
// - Practical limit: 50,000-500,000 connections (hardware dependent)

// IOCP based
// - Similar to epoll performance characteristics
// - Optimized for Windows kernel
// - Can handle 100,000+ connections efficiently
```

#### Latency Characteristics
```c
// Typical latency measurements (microseconds) for echo server:

// Single Connection:
// - Blocking I/O:     50-100 μs
// - Non-blocking I/O: 30-80 μs  
// - select():         40-90 μs
// - poll():           35-85 μs
// - epoll():          25-60 μs
// - kqueue():         20-55 μs
// - IOCP:             30-70 μs

// 1,000 Concurrent Connections:
// - Blocking I/O:     500-2000 μs (high variation due to context switching)
// - Non-blocking I/O: 200-800 μs
// - select():         1000-5000 μs (becomes inefficient)
// - poll():           300-1200 μs
// - epoll():          50-150 μs
// - kqueue():         45-140 μs
// - IOCP:             60-180 μs

// 10,000 Concurrent Connections:
// - Blocking I/O:     Not practical (resource exhaustion)
// - Non-blocking I/O: 2000-10000 μs (polling overhead)
// - select():         Not practical (FD_SETSIZE limit)
// - poll():           5000-20000 μs (quadratic scaling)
// - epoll():          80-250 μs
// - kqueue():         70-230 μs  
// - IOCP:             100-300 μs
```

### Resource Usage Analysis

#### Memory Usage Patterns
```c
// Memory usage breakdown for 10,000 concurrent connections:

// Blocking I/O (thread-per-connection):
// - Thread stacks: 10,000 × 8MB = 80GB (impossible)
// - Connection data: 10,000 × 4KB = 40MB
// - Total: Not feasible due to memory constraints

// Event-driven I/O (single-threaded):
// - Application buffers: 10,000 × 8KB = 80MB
// - Kernel structures: 10,000 × 1KB = 10MB  
// - Event system overhead: 5-10MB
// - Total: ~100MB (very manageable)

// Multi-threaded event-driven (4 worker threads):
// - Thread stacks: 4 × 8MB = 32MB
// - Application buffers: 10,000 × 8KB = 80MB
// - Kernel structures: 10,000 × 1KB = 10MB
// - Event system overhead: 10-15MB
// - Total: ~140MB (still very reasonable)
```

#### CPU Usage Patterns
```c
// CPU usage for handling 1,000 requests/second:

// Blocking I/O:
// - Context switching overhead: 30-50%
// - Actual work: 20-30%
// - System calls: 20-30%
// - Idle/waiting: 10-20%

// Non-blocking I/O with polling:
// - Polling overhead: 40-60%
// - Actual work: 25-35%
// - System calls: 10-15%
// - Idle: 5-10%

// Event-driven I/O:
// - Event handling: 10-20%
// - Actual work: 40-60%
// - System calls: 15-25%
// - Idle: 10-30%
```

### When to Use Each Model

#### Blocking I/O - Best For:
```c
// 1. Simple client applications
void simple_client_example() {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    connect(sockfd, ...);
    
    // Simple request-response, blocking is fine
    send(sockfd, request, request_len, 0);
    recv(sockfd, response, response_len, 0);
    
    close(sockfd);
}

// 2. Low-concurrency servers (< 50 clients)
// 3. Prototype development where simplicity matters
// 4. Educational purposes to understand basics
// 5. Applications with predictable, short-lived connections
```

#### Non-blocking I/O - Best For:
```c
// 1. Interactive applications that need responsiveness
void interactive_client() {
    make_socket_non_blocking(sockfd);
    
    while (running) {
        // Handle UI events
        process_user_input();
        
        // Check for network data without blocking
        if (data_available(sockfd)) {
            handle_network_data();
        }
        
        // Update display
        refresh_ui();
    }
}

// 2. Applications that mix network I/O with other tasks
// 3. Real-time systems where blocking is unacceptable
// 4. Single-threaded applications handling multiple connections
```

#### select() - Best For:
```c
// 1. Portable applications (works everywhere)
// 2. Small to medium scale servers (< 1000 connections)
// 3. Applications monitoring mixed file descriptor types
void mixed_fd_monitoring() {
    fd_set readfds;
    FD_ZERO(&readfds);
    
    // Monitor sockets
    FD_SET(tcp_socket, &readfds);
    FD_SET(udp_socket, &readfds);
    
    // Monitor files  
    FD_SET(log_file, &readfds);
    
    // Monitor pipes
    FD_SET(pipe_fd, &readfds);
    
    select(max_fd + 1, &readfds, NULL, NULL, &timeout);
}

// 4. Legacy systems where other options aren't available
```

#### poll() - Best For:
```c
// 1. Systems without epoll/kqueue but need > 1024 FDs
// 2. Applications requiring fine-grained event control
void poll_with_custom_events() {
    struct pollfd fds[NUM_CLIENTS];
    
    for (int i = 0; i < num_clients; i++) {
        fds[i].events = POLLIN;
        
        // Add custom events based on client state
        if (clients[i].has_pending_data) {
            fds[i].events |= POLLOUT;
        }
        
        if (clients[i].expecting_urgent_data) {
            fds[i].events |= POLLPRI;
        }
    }
    
    poll(fds, num_clients, timeout);
}

// 3. Moderate scale servers (1,000-10,000 connections)
// 4. Applications needing better portability than epoll
```

#### epoll/kqueue - Best For:
```c
// 1. High-performance servers (10,000+ connections)
// 2. Real-time applications requiring low latency
// 3. Systems with high connection churn
void high_performance_server() {
    // Can efficiently handle massive connection counts
    epoll_server(server_fd); // 100,000+ connections
}

// 4. Applications requiring maximum scalability
// 5. Load balancers, proxies, CDN edge servers
// 6. Game servers, chat servers, IoT gateways
```

#### IOCP - Best For:
```c
// 1. Windows-based high-performance applications
// 2. Applications requiring completion-based notifications
// 3. Systems with heavy I/O workloads
void windows_high_performance_server() {
    // Optimized for Windows kernel
    iocp_server(server_socket); // 50,000+ connections
}

// 4. File servers with large file transfers
// 5. Database servers on Windows
// 6. Enterprise applications requiring Windows integration
```

### Decision Matrix

**Choose your I/O model based on these criteria:**

```c
// Decision tree for I/O model selection:

if (platform == WINDOWS && connections > 1000) {
    use_iocp();
} else if (platform == LINUX && connections > 1000) {
    use_epoll();
} else if (platform == BSD_OR_MACOS && connections > 1000) {
    use_kqueue();
} else if (connections > 100 && connections < 1000) {
    use_poll();
} else if (connections < 100 && portability_important) {
    use_select();
} else if (connections < 50 && simplicity_important) {
    use_blocking_io_with_threads();
} else {
    // Default fallback
    use_poll();
}
```

**Performance vs Complexity Trade-off:**

```
High Performance ↑
                 │
    epoll/kqueue │ ████████████████ │ High Complexity
            IOCP │ ███████████████  │
            poll │ ████████         │
          select │ ██████           │
   Non-blocking  │ ████             │
      Blocking   │ ██               │ Low Complexity
                 └─────────────────→
                              Simple
```

## Performance Considerations and Optimization

### Understanding the C10K Problem

The **C10K Problem** refers to the challenge of handling 10,000 concurrent connections on a single server. This problem highlighted the limitations of traditional I/O models and drove the development of modern event-driven architectures.

#### Historical Context
```c
// Pre-2000s: Thread-per-connection model
// Problem: 10,000 threads × 8MB stack = 80GB RAM (impossible)

void old_school_server() {
    while (1) {
        int client = accept(server_fd, ...);
        pthread_t thread;
        pthread_create(&thread, NULL, handle_client, &client);
        // Each thread uses ~8MB of stack space
        // Context switching becomes expensive with many threads
    }
}

// Modern solution: Event-driven architecture
void modern_server() {
    int epoll_fd = epoll_create1(EPOLL_CLOEXEC);
    
    while (1) {
        struct epoll_event events[MAX_EVENTS];
        int nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);
        
        // Single thread handles all connections efficiently
        for (int i = 0; i < nfds; i++) {
            handle_event(&events[i]);
        }
    }
}
```

### Optimization Strategies

#### 1. Buffer Management Optimization
```c
// Poor buffer management (creates memory pressure):
void inefficient_buffer_handling(int client_fd) {
    char* buffer = malloc(64 * 1024); // 64KB per client
    recv(client_fd, buffer, 64 * 1024, 0);
    // Process data...
    free(buffer); // Frequent malloc/free causes fragmentation
}

// Optimized buffer management:
typedef struct {
    char* data;
    size_t capacity;
    size_t length;
    size_t read_pos;
} buffer_pool_t;

// Pre-allocated buffer pool
static buffer_pool_t* buffer_pool = NULL;
static size_t pool_size = 0;

buffer_pool_t* get_buffer_from_pool() {
    for (size_t i = 0; i < pool_size; i++) {
        if (buffer_pool[i].data == NULL) {
            buffer_pool[i].data = malloc(8192); // 8KB buffers
            buffer_pool[i].capacity = 8192;
            buffer_pool[i].length = 0;
            buffer_pool[i].read_pos = 0;
            return &buffer_pool[i];
        }
    }
    return NULL; // Pool exhausted
}

void return_buffer_to_pool(buffer_pool_t* buf) {
    buf->length = 0;
    buf->read_pos = 0;
    // Keep allocated memory for reuse
}

// Even better: Ring buffer for streaming data
typedef struct {
    char* data;
    size_t capacity;
    size_t head;  // Write position
    size_t tail;  // Read position
    size_t count; // Current data count
} ring_buffer_t;

size_t ring_buffer_write(ring_buffer_t* rb, const void* data, size_t len) {
    size_t available = rb->capacity - rb->count;
    if (len > available) {
        len = available; // Write what we can
    }
    
    size_t first_chunk = rb->capacity - rb->head;
    if (first_chunk >= len) {
        // No wraparound needed
        memcpy(rb->data + rb->head, data, len);
    } else {
        // Need to wrap around
        memcpy(rb->data + rb->head, data, first_chunk);
        memcpy(rb->data, (char*)data + first_chunk, len - first_chunk);
    }
    
    rb->head = (rb->head + len) % rb->capacity;
    rb->count += len;
    return len;
}
```

#### 2. Connection Pooling and Reuse
```c
// Connection pooling for backend connections (e.g., database, cache)
typedef struct {
    int fd;
    time_t last_used;
    int in_use;
    char server_addr[256];
} pooled_connection_t;

typedef struct {
    pooled_connection_t* connections;
    size_t pool_size;
    size_t active_count;
    pthread_mutex_t mutex;
} connection_pool_t;

pooled_connection_t* get_connection_from_pool(connection_pool_t* pool, const char* server) {
    pthread_mutex_lock(&pool->mutex);
    
    // Look for existing idle connection to same server
    for (size_t i = 0; i < pool->pool_size; i++) {
        if (!pool->connections[i].in_use && 
            strcmp(pool->connections[i].server_addr, server) == 0) {
            
            // Check if connection is still alive
            if (is_connection_alive(pool->connections[i].fd)) {
                pool->connections[i].in_use = 1;
                pool->connections[i].last_used = time(NULL);
                pthread_mutex_unlock(&pool->mutex);
                return &pool->connections[i];
            } else {
                // Connection is dead, clean it up
                close(pool->connections[i].fd);
                pool->connections[i].fd = -1;
            }
        }
    }
    
    // No available connection, create new one if pool has space
    for (size_t i = 0; i < pool->pool_size; i++) {
        if (pool->connections[i].fd == -1) {
            int new_fd = create_connection_to_server(server);
            if (new_fd >= 0) {
                pool->connections[i].fd = new_fd;
                pool->connections[i].in_use = 1;
                pool->connections[i].last_used = time(NULL);
                strncpy(pool->connections[i].server_addr, server, 255);
                pool->active_count++;
                pthread_mutex_unlock(&pool->mutex);
                return &pool->connections[i];
            }
        }
    }
    
    pthread_mutex_unlock(&pool->mutex);
    return NULL; // Pool is full
}

void return_connection_to_pool(connection_pool_t* pool, pooled_connection_t* conn) {
    pthread_mutex_lock(&pool->mutex);
    conn->in_use = 0;
    conn->last_used = time(NULL);
    pthread_mutex_unlock(&pool->mutex);
}
```

#### 3. Zero-Copy Techniques
```c
// Traditional copy-heavy approach:
void traditional_file_serving(int client_fd, const char* filename) {
    FILE* file = fopen(filename, "rb");
    char buffer[8192];
    size_t bytes_read;
    
    while ((bytes_read = fread(buffer, 1, sizeof(buffer), file)) > 0) {
        send(client_fd, buffer, bytes_read, 0);
        // Data copied: file -> buffer -> kernel -> network
        // Multiple copies waste CPU and memory bandwidth
    }
    
    fclose(file);
}

// Zero-copy with sendfile() (Linux):
#ifdef __linux__
#include <sys/sendfile.h>

void zero_copy_file_serving(int client_fd, const char* filename) {
    int file_fd = open(filename, O_RDONLY);
    if (file_fd < 0) return;
    
    struct stat file_stat;
    if (fstat(file_fd, &file_stat) < 0) {
        close(file_fd);
        return;
    }
    
    // Zero-copy transfer: file -> network (no user-space copying)
    off_t offset = 0;
    ssize_t sent = sendfile(client_fd, file_fd, &offset, file_stat.st_size);
    
    close(file_fd);
}
#endif

// Alternative: Memory-mapped files
void mmap_file_serving(int client_fd, const char* filename) {
    int file_fd = open(filename, O_RDONLY);
    struct stat file_stat;
    fstat(file_fd, &file_stat);
    
    // Map file into memory
    void* file_data = mmap(NULL, file_stat.st_size, PROT_READ, MAP_PRIVATE, file_fd, 0);
    if (file_data != MAP_FAILED) {
        // Send mapped memory directly
        send(client_fd, file_data, file_stat.st_size, 0);
        munmap(file_data, file_stat.st_size);
    }
    
    close(file_fd);
}
```

#### 4. CPU Cache Optimization
```c
// Cache-friendly data structures:
// BAD: Array of structures (poor cache locality)
typedef struct {
    int fd;
    char buffer[4096];
    time_t last_activity;
    int state;
} client_bad_t;

client_bad_t clients[10000]; // Buffer data scattered in memory

// GOOD: Structure of arrays (better cache locality)
typedef struct {
    int* fds;                    // All file descriptors together
    time_t* last_activities;     // All timestamps together  
    int* states;                 // All states together
    char* buffers;               // All buffers together
} client_pool_good_t;

// When iterating through active connections, we only touch the arrays we need
void process_active_connections(client_pool_good_t* pool, int count) {
    time_t now = time(NULL);
    
    // This loop has excellent cache locality
    for (int i = 0; i < count; i++) {
        if (now - pool->last_activities[i] > TIMEOUT) {
            close(pool->fds[i]);
            // Mark as inactive...
        }
    }
}
```

#### 5. NUMA Awareness (Multi-socket systems)
```c
// NUMA-aware thread binding
#ifdef __linux__
#include <numa.h>
#include <sched.h>

void setup_numa_aware_workers(int num_workers) {
    int num_nodes = numa_max_node() + 1;
    
    for (int i = 0; i < num_workers; i++) {
        pthread_t worker;
        worker_context_t* ctx = malloc(sizeof(worker_context_t));
        ctx->worker_id = i;
        ctx->numa_node = i % num_nodes;
        
        pthread_create(&worker, NULL, numa_aware_worker, ctx);
        
        // Bind thread to specific NUMA node
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        
        // Get CPUs for this NUMA node
        struct bitmask* node_cpus = numa_allocate_cpumask();
        numa_node_to_cpus(ctx->numa_node, node_cpus);
        
        for (int cpu = 0; cpu < numa_num_possible_cpus(); cpu++) {
            if (numa_bitmask_isbitset(node_cpus, cpu)) {
                CPU_SET(cpu, &cpuset);
            }
        }
        
        pthread_setaffinity_np(worker, sizeof(cpuset), &cpuset);
        numa_free_cpumask(node_cpus);
    }
}

void* numa_aware_worker(void* arg) {
    worker_context_t* ctx = (worker_context_t*)arg;
    
    // Allocate memory on local NUMA node
    numa_set_preferred(ctx->numa_node);
    
    // Worker event loop...
    return NULL;
}
#endif
```

### Benchmarking and Profiling

#### Performance Measurement Tools
```c
// Built-in performance counters
typedef struct {
    atomic_long connections_accepted;
    atomic_long connections_closed;
    atomic_long bytes_received;
    atomic_long bytes_sent;
    atomic_long messages_processed;
    
    // Latency tracking
    atomic_long total_response_time_us;
    atomic_long min_response_time_us;
    atomic_long max_response_time_us;
    
    // Error counters
    atomic_long connection_errors;
    atomic_long send_errors;
    atomic_long recv_errors;
} performance_counters_t;

void update_response_time(performance_counters_t* counters, long response_time_us) {
    atomic_fetch_add(&counters->total_response_time_us, response_time_us);
    
    // Update min (using compare-and-swap loop)
    long current_min = atomic_load(&counters->min_response_time_us);
    while (response_time_us < current_min) {
        if (atomic_compare_exchange_weak(&counters->min_response_time_us, 
                                       &current_min, response_time_us)) {
            break;
        }
    }
    
    // Update max
    long current_max = atomic_load(&counters->max_response_time_us);
    while (response_time_us > current_max) {
        if (atomic_compare_exchange_weak(&counters->max_response_time_us, 
                                       &current_max, response_time_us)) {
            break;
        }
    }
}

// Real-time statistics reporting
void* stats_reporter_thread(void* arg) {
    performance_counters_t* counters = (performance_counters_t*)arg;
    
    while (running) {
        sleep(5); // Report every 5 seconds
        
        long connections = atomic_load(&counters->connections_accepted);
        long bytes_rx = atomic_load(&counters->bytes_received);
        long bytes_tx = atomic_load(&counters->bytes_sent);
        long messages = atomic_load(&counters->messages_processed);
        
        printf("Stats: %ld connections, %ld msgs, %ld bytes RX, %ld bytes TX\n",
               connections, messages, bytes_rx, bytes_tx);
        
        if (messages > 0) {
            long avg_latency = atomic_load(&counters->total_response_time_us) / messages;
            long min_latency = atomic_load(&counters->min_response_time_us);
            long max_latency = atomic_load(&counters->max_response_time_us);
            
            printf("Latency: avg=%ld μs, min=%ld μs, max=%ld μs\n",
                   avg_latency, min_latency, max_latency);
        }
    }
    
    return NULL;
}
```

#### System-level Performance Monitoring
```bash
# CPU usage and context switches
top -H -p <server_pid>          # Show threads
vmstat 1                        # Context switches per second
perf stat -p <server_pid>       # Detailed CPU counters

# Memory usage
valgrind --tool=massif ./server # Memory profiling
pmap -x <server_pid>            # Memory mapping
cat /proc/<server_pid>/status   # Process memory info

# Network performance
ss -tulpn                       # Show listening sockets
netstat -i                      # Network interface statistics
iftop                          # Real-time network usage
tcpdump -i any port 8080       # Network packet capture

# I/O performance
iostat -x 1                    # Disk I/O statistics
iotop                          # I/O usage by process
strace -p <server_pid>         # System call tracing

# Advanced profiling
perf record -g ./server        # CPU profiling with call graphs
perf report                    # View profiling results
gprof ./server gmon.out        # Function-level profiling (if compiled with -pg)
```

#### Load Testing Tools and Techniques
```bash
# Apache Bench (simple HTTP testing)
ab -n 100000 -c 1000 http://localhost:8080/

# wrk (modern HTTP benchmark tool)
wrk -t12 -c1000 -d30s --timeout 2s http://localhost:8080/

# Custom TCP load tester
./tcp_flood_test --host localhost --port 8080 --connections 10000 --duration 60

# Sustained load testing
while true; do
    echo "Starting load test batch..."
    wrk -t8 -c500 -d60s http://localhost:8080/
    sleep 5
done

# Memory leak detection under load
valgrind --tool=memcheck --leak-check=full ./server &
SERVER_PID=$!
sleep 2
wrk -t4 -c100 -d300s http://localhost:8080/  # 5 minutes of load
kill $SERVER_PID
```

### Optimization Checklist

#### ✅ Application Level
- [ ] Use appropriate I/O model for your scale requirements
- [ ] Implement connection pooling for backend services
- [ ] Use buffer pools to reduce memory allocation overhead
- [ ] Implement zero-copy techniques for large data transfers
- [ ] Cache frequently accessed data in memory
- [ ] Use lock-free data structures where possible
- [ ] Profile your application regularly

#### ✅ System Level
- [ ] Tune kernel parameters (net.core.somaxconn, net.core.rmem_max, etc.)
- [ ] Use appropriate TCP congestion control algorithm
- [ ] Configure firewall rules efficiently
- [ ] Optimize file descriptor limits (ulimit -n)
- [ ] Use dedicated network interfaces for high traffic
- [ ] Consider NUMA topology for multi-socket systems
- [ ] Monitor system resources continuously

#### ✅ Network Level
- [ ] Use appropriate network buffer sizes
- [ ] Configure TCP_NODELAY for low-latency applications
- [ ] Set SO_REUSEADDR and SO_REUSEPORT appropriately
- [ ] Consider using UDP for appropriate use cases
- [ ] Implement proper backpressure handling
- [ ] Use connection multiplexing when beneficial
- [ ] Monitor network saturation and packet loss

## Assessment Checklist

### Theoretical Understanding
- [ ] **Explain the fundamental differences** between blocking and non-blocking I/O
- [ ] **Describe the C10K problem** and why traditional threading models fail at scale
- [ ] **Compare Level-triggered vs Edge-triggered** event notification mechanisms
- [ ] **Analyze memory and CPU usage patterns** for different I/O models
- [ ] **Identify appropriate use cases** for each I/O model based on requirements

### Practical Implementation Skills
- [ ] **Implement a blocking I/O server** with proper timeout handling and error management
- [ ] **Create a non-blocking I/O application** that handles EAGAIN/EWOULDBLOCK correctly
- [ ] **Build a select()-based server** that monitors multiple file descriptors simultaneously
- [ ] **Develop a poll()-based server** with dynamic client management
- [ ] **Create an epoll-based high-performance server** handling 10,000+ concurrent connections
- [ ] **Implement proper error handling** for all I/O operations and edge cases

### System Integration Knowledge
- [ ] **Configure system parameters** for high-performance networking (ulimits, kernel parameters)
- [ ] **Use profiling tools** to identify performance bottlenecks in I/O-intensive applications
- [ ] **Implement connection pooling** and resource management strategies
- [ ] **Apply zero-copy techniques** for efficient data transfer
- [ ] **Design NUMA-aware applications** for multi-socket systems
- [ ] **Create cross-platform abstractions** that work efficiently on different operating systems

### Performance Analysis
- [ ] **Measure and compare performance** of different I/O models under various load conditions
- [ ] **Identify scalability bottlenecks** and implement appropriate solutions
- [ ] **Calculate resource requirements** for target concurrent connection counts
- [ ] **Design load testing strategies** to validate application performance
- [ ] **Implement monitoring and alerting** for production systems

### Advanced Topics
- [ ] **Implement state machines** for complex protocol handling
- [ ] **Design event-driven architectures** with proper separation of concerns
- [ ] **Create thread pools** with work stealing for CPU-intensive tasks
- [ ] **Implement backpressure mechanisms** to handle overload conditions
- [ ] **Build distributed systems** using event-driven I/O as foundation

### Real-world Application
- [ ] **Design a chat server** supporting thousands of concurrent users
- [ ] **Build an HTTP load balancer** with health checking and failover
- [ ] **Create a proxy server** with connection pooling and request routing
- [ ] **Implement a file server** with zero-copy transfers and caching
- [ ] **Design an IoT gateway** handling sensor data from thousands of devices

## Next Steps

### Immediate Follow-up Topics
After mastering socket I/O models, explore these related areas:

1. **Asynchronous Programming Models**
   - async/await patterns in modern languages (C++20 coroutines, Rust async, Node.js)
   - Promise-based architectures
   - Actor model implementations

2. **High-Performance Networking Libraries**
   - **libuv**: Cross-platform asynchronous I/O library (used by Node.js)
   - **Boost.Asio**: C++ asynchronous I/O library
   - **Netty**: Java NIO framework
   - **Tokio**: Rust asynchronous runtime

3. **Advanced Networking Protocols**
   - HTTP/2 and HTTP/3 multiplexing
   - WebSocket full-duplex communication
   - QUIC protocol implementation
   - gRPC streaming

4. **Distributed Systems Patterns**
   - Service mesh architectures
   - Event sourcing and CQRS
   - Reactive streaming
   - Circuit breaker patterns

### Recommended Projects

**Beginner Projects:**
1. **Multi-protocol server** handling HTTP, WebSocket, and custom protocols
2. **Connection pooling library** with health checking and load balancing
3. **Simple chat application** with rooms and user management

**Intermediate Projects:**
1. **HTTP reverse proxy** with SSL termination and request routing
2. **Real-time game server** handling thousands of players
3. **Pub/sub message broker** with topic-based routing

**Advanced Projects:**
1. **CDN edge server** with intelligent caching and content delivery
2. **Database connection pooler** (like PgBouncer) with connection multiplexing
3. **IoT data ingestion platform** handling millions of sensor updates

### Career Development

**Roles that heavily use these skills:**
- **Backend/Systems Engineer**: Building scalable server applications
- **Site Reliability Engineer (SRE)**: Optimizing system performance and reliability  
- **Network Software Engineer**: Developing networking infrastructure
- **Game Server Developer**: Creating real-time multiplayer experiences
- **IoT Platform Engineer**: Building device communication systems
- **Performance Engineer**: Optimizing application and system performance

**Industry Applications:**
- **Social Media Platforms**: Handling millions of concurrent users
- **Financial Trading Systems**: Ultra-low latency requirement systems
- **Streaming Services**: Real-time video/audio delivery
- **Online Gaming**: Multiplayer game servers
- **IoT Platforms**: Device management and data collection
- **CDN Services**: Content delivery and caching

## Resources

### Essential Reading
- **"UNIX Network Programming, Volume 1"** by W. Richard Stevens (Chapter 6: I/O Multiplexing)
- **"The C10K Problem"** by Dan Kegel - Foundational paper on high-concurrency servers
- **"High Performance Browser Networking"** by Ilya Grigorik - Modern networking concepts
- **"Systems Performance"** by Brendan Gregg - Performance analysis and optimization

### Online Resources
- **Linux man pages**: select(2), poll(2), epoll(7), kqueue(2)
- **"Scalable Network Programming"** - IBM Developer tutorials
- **"The Architecture of Open Source Applications"** - nginx and other high-performance servers
- **Kernel documentation**: Documentation/networking/ in Linux kernel source

### Development Tools
```bash
# Performance profiling
sudo apt-get install linux-tools-generic  # perf
sudo apt-get install valgrind             # Memory profiling
sudo apt-get install strace               # System call tracing

# Network tools
sudo apt-get install tcpdump wireshark    # Packet analysis
sudo apt-get install iftop netstat ss     # Network monitoring
sudo apt-get install wrk apache2-utils    # Load testing

# Development
sudo apt-get install build-essential      # Compilers and build tools
sudo apt-get install gdb                  # Debugging
sudo apt-get install cmake                # Build system
```

### Example Code Repositories
- **libuv**: Cross-platform async I/O library
- **libevent**: Event notification library
- **nginx**: High-performance web server source code
- **Redis**: In-memory data structure store (excellent example of event-driven architecture)
- **HAProxy**: Load balancer and proxy server

### Community and Support
- **Stack Overflow**: networking, performance, unix tags
- **Reddit**: r/networking, r/systems, r/programming
- **Linux kernel mailing list**: For deep kernel networking discussions
- **High Scalability blog**: Real-world architecture case studies

### Certification and Formal Education
- **Linux Foundation**: Networking and performance courses
- **Coursera/edX**: Computer Networks and Distributed Systems courses
- **AWS/GCP/Azure**: Cloud networking and performance optimization
- **CNCF**: Cloud-native networking technologies

This comprehensive foundation in socket I/O models will serve as a cornerstone for advanced networking and systems programming throughout your career. The concepts and techniques learned here apply broadly across many domains and technologies.

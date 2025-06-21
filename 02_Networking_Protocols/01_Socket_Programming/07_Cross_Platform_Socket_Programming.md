# Cross-Platform Socket Programming

*Last Updated: June 21, 2025*

## Overview

This module covers cross-platform socket programming, focusing on writing portable network code that works across Windows, Linux, macOS, and other operating systems. You'll learn platform-specific differences, abstraction techniques, and best practices for maintaining a single codebase.

## Learning Objectives

By the end of this module, you should be able to:
- Understand differences between Windows Winsock and POSIX sockets
- Write portable socket code using abstraction layers
- Handle platform-specific considerations effectively
- Manage error codes and differences across platforms
- Build cross-platform network applications

## Topics Covered

### 1. Windows Socket API (Winsock)
- Winsock initialization and cleanup
- Windows-specific socket functions
- IOCP (I/O Completion Ports) on Windows

### 2. POSIX Socket API
- Unix/Linux socket implementation
- BSD socket heritage
- Platform-specific extensions

### 3. Abstraction Layers
- Creating portable socket interfaces
- Handling platform differences
- Configuration management

### 4. Platform-specific Considerations
- Error handling differences
- Data type variations
- Build system integration

### 5. Cross-platform Build Systems
- CMake for socket applications
- Conditional compilation
- Dependency management

## Practical Exercises

1. **Platform Detection System**
   - Create macros for platform detection
   - Implement platform-specific code paths

2. **Portable Socket Library**
   - Build abstraction layer for common operations
   - Test on multiple platforms

3. **Cross-platform Server**
   - Implement high-performance server
   - Use best I/O model for each platform

4. **Build System Setup**
   - Configure CMake for cross-platform builds
   - Handle platform-specific dependencies

## Code Examples

### 1. Platform Detection and Basic Setup

```c
// platform.h - Platform detection and basic definitions
#ifndef PLATFORM_H
#define PLATFORM_H

// Platform detection
#ifdef _WIN32
    #define PLATFORM_WINDOWS
    #ifdef _WIN64
        #define PLATFORM_WINDOWS_64
    #else
        #define PLATFORM_WINDOWS_32
    #endif
#elif defined(__linux__)
    #define PLATFORM_LINUX
#elif defined(__APPLE__)
    #define PLATFORM_MACOS
#elif defined(__unix__)
    #define PLATFORM_UNIX
#else
    #error "Unknown platform"
#endif

// Windows-specific includes and definitions
#ifdef PLATFORM_WINDOWS
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #include <mswsock.h>
    
    // Link with Winsock library
    #pragma comment(lib, "ws2_32.lib")
    #pragma comment(lib, "mswsock.lib")
    
    // Windows socket types and constants
    typedef SOCKET socket_t;
    typedef int socklen_t;
    
    #define INVALID_SOCKET_FD INVALID_SOCKET
    #define SOCKET_ERROR_CODE WSAGetLastError()
    #define CLOSE_SOCKET(s) closesocket(s)
    
    // Windows doesn't have these POSIX constants
    #ifndef SHUT_RD
        #define SHUT_RD SD_RECEIVE
        #define SHUT_WR SD_SEND
        #define SHUT_RDWR SD_BOTH
    #endif
    
#else
    // POSIX includes
    #include <sys/socket.h>
    #include <sys/types.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <errno.h>
    
    // POSIX socket types and constants
    typedef int socket_t;
    
    #define INVALID_SOCKET_FD (-1)
    #define SOCKET_ERROR (-1)
    #define SOCKET_ERROR_CODE errno
    #define CLOSE_SOCKET(s) close(s)
    
    // Platform-specific I/O multiplexing
    #ifdef PLATFORM_LINUX
        #include <sys/epoll.h>
    #elif defined(PLATFORM_MACOS) || defined(PLATFORM_UNIX)
        #include <sys/event.h>
    #endif
#endif

// Common definitions
#define SOCKET_SUCCESS 0

#endif // PLATFORM_H
```

### 2. Socket Initialization and Cleanup

```c
// socket_init.c - Platform-specific initialization
#include "platform.h"
#include <stdio.h>

int socket_library_init(void) {
#ifdef PLATFORM_WINDOWS
    WSADATA wsaData;
    int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (result != 0) {
        fprintf(stderr, "WSAStartup failed: %d\n", result);
        return -1;
    }
    
    // Verify Winsock version
    if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2) {
        fprintf(stderr, "Could not find a usable version of Winsock.dll\n");
        WSACleanup();
        return -1;
    }
    
    printf("Winsock 2.2 initialized successfully\n");
#endif
    return 0;
}

void socket_library_cleanup(void) {
#ifdef PLATFORM_WINDOWS
    WSACleanup();
    printf("Winsock cleanup completed\n");
#endif
}

// Get last socket error in a portable way
int get_socket_error(void) {
    return SOCKET_ERROR_CODE;
}

// Convert socket error to string
const char* socket_error_string(int error_code) {
#ifdef PLATFORM_WINDOWS
    static char error_buffer[256];
    FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                  NULL, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                  error_buffer, sizeof(error_buffer), NULL);
    return error_buffer;
#else
    return strerror(error_code);
#endif
}
```

### 3. Portable Socket Operations

```c
// socket_ops.c - Portable socket operations
#include "platform.h"
#include <stdio.h>
#include <string.h>

socket_t create_tcp_socket(void) {
    socket_t sockfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sockfd == INVALID_SOCKET_FD) {
        fprintf(stderr, "Socket creation failed: %s\n", 
                socket_error_string(get_socket_error()));
        return INVALID_SOCKET_FD;
    }
    return sockfd;
}

socket_t create_udp_socket(void) {
    socket_t sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sockfd == INVALID_SOCKET_FD) {
        fprintf(stderr, "UDP socket creation failed: %s\n", 
                socket_error_string(get_socket_error()));
        return INVALID_SOCKET_FD;
    }
    return sockfd;
}

int set_socket_nonblocking(socket_t sockfd) {
#ifdef PLATFORM_WINDOWS
    u_long mode = 1;  // 1 to enable non-blocking socket
    if (ioctlsocket(sockfd, FIONBIO, &mode) != 0) {
        fprintf(stderr, "ioctlsocket failed: %s\n", 
                socket_error_string(get_socket_error()));
        return -1;
    }
#else
    int flags = fcntl(sockfd, F_GETFL, 0);
    if (flags < 0) {
        perror("fcntl F_GETFL");
        return -1;
    }
    
    if (fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) < 0) {
        perror("fcntl F_SETFL O_NONBLOCK");
        return -1;
    }
#endif
    return 0;
}

int set_socket_reuse_addr(socket_t sockfd) {
    int reuse = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, 
                   (const char*)&reuse, sizeof(reuse)) < 0) {
        fprintf(stderr, "setsockopt SO_REUSEADDR failed: %s\n", 
                socket_error_string(get_socket_error()));
        return -1;
    }
    return 0;
}

int bind_socket(socket_t sockfd, const char* ip, int port) {
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    
    if (ip == NULL || strcmp(ip, "0.0.0.0") == 0) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        addr.sin_addr.s_addr = inet_addr(ip);
        if (addr.sin_addr.s_addr == INADDR_NONE) {
            fprintf(stderr, "Invalid IP address: %s\n", ip);
            return -1;
        }
    }
    
    if (bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "Bind failed: %s\n", 
                socket_error_string(get_socket_error()));
        return -1;
    }
    
    return 0;
}

int connect_socket(socket_t sockfd, const char* ip, int port) {
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr(ip);
    
    if (addr.sin_addr.s_addr == INADDR_NONE) {
        fprintf(stderr, "Invalid IP address: %s\n", ip);
        return -1;
    }
    
    if (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        int error = get_socket_error();
#ifdef PLATFORM_WINDOWS
        if (error != WSAEWOULDBLOCK) {
#else
        if (error != EINPROGRESS) {
#endif
            fprintf(stderr, "Connect failed: %s\n", socket_error_string(error));
            return -1;
        }
        // Non-blocking connect in progress
        return 1;
    }
    
    return 0;  // Connected immediately
}

void close_socket(socket_t sockfd) {
    if (sockfd != INVALID_SOCKET_FD) {
        CLOSE_SOCKET(sockfd);
    }
}
```

### 4. Cross-platform I/O Multiplexing

```c
// io_multiplex.h - Cross-platform I/O multiplexing interface
#ifndef IO_MULTIPLEX_H
#define IO_MULTIPLEX_H

#include "platform.h"

// Event types
#define IO_EVENT_READ  0x01
#define IO_EVENT_WRITE 0x02
#define IO_EVENT_ERROR 0x04

typedef struct {
    socket_t fd;
    int events;     // Requested events
    int revents;    // Returned events
    void* user_data;
} io_event_t;

typedef struct io_context io_context_t;

// Interface functions
io_context_t* io_context_create(int max_events);
void io_context_destroy(io_context_t* ctx);
int io_context_add(io_context_t* ctx, socket_t fd, int events, void* user_data);
int io_context_remove(io_context_t* ctx, socket_t fd);
int io_context_wait(io_context_t* ctx, io_event_t* events, int max_events, int timeout_ms);

#endif // IO_MULTIPLEX_H
```

```c
// io_multiplex.c - Cross-platform I/O multiplexing implementation
#include "io_multiplex.h"
#include <stdlib.h>
#include <string.h>

#ifdef PLATFORM_WINDOWS
// Windows IOCP implementation
struct io_context {
    HANDLE completion_port;
    int max_events;
};

io_context_t* io_context_create(int max_events) {
    io_context_t* ctx = malloc(sizeof(io_context_t));
    if (!ctx) return NULL;
    
    ctx->completion_port = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, 0);
    if (ctx->completion_port == NULL) {
        free(ctx);
        return NULL;
    }
    
    ctx->max_events = max_events;
    return ctx;
}

void io_context_destroy(io_context_t* ctx) {
    if (ctx) {
        CloseHandle(ctx->completion_port);
        free(ctx);
    }
}

int io_context_add(io_context_t* ctx, socket_t fd, int events, void* user_data) {
    // Associate socket with completion port
    if (CreateIoCompletionPort((HANDLE)fd, ctx->completion_port, (ULONG_PTR)user_data, 0) == NULL) {
        return -1;
    }
    return 0;
}

int io_context_remove(io_context_t* ctx, socket_t fd) {
    // Windows doesn't have a direct way to remove from IOCP
    // Usually handled by closing the socket
    return 0;
}

int io_context_wait(io_context_t* ctx, io_event_t* events, int max_events, int timeout_ms) {
    DWORD bytes_transferred;
    ULONG_PTR completion_key;
    OVERLAPPED* overlapped;
    
    BOOL result = GetQueuedCompletionStatus(ctx->completion_port,
                                          &bytes_transferred,
                                          &completion_key,
                                          &overlapped,
                                          timeout_ms == -1 ? INFINITE : timeout_ms);
    
    if (result || overlapped != NULL) {
        events[0].fd = (socket_t)completion_key;  // Simplified
        events[0].revents = IO_EVENT_READ;        // Simplified
        events[0].user_data = (void*)completion_key;
        return 1;
    }
    
    return 0;  // Timeout or error
}

#elif defined(PLATFORM_LINUX)
// Linux epoll implementation
#include <sys/epoll.h>

struct io_context {
    int epoll_fd;
    int max_events;
};

io_context_t* io_context_create(int max_events) {
    io_context_t* ctx = malloc(sizeof(io_context_t));
    if (!ctx) return NULL;
    
    ctx->epoll_fd = epoll_create1(EPOLL_CLOEXEC);
    if (ctx->epoll_fd < 0) {
        free(ctx);
        return NULL;
    }
    
    ctx->max_events = max_events;
    return ctx;
}

void io_context_destroy(io_context_t* ctx) {
    if (ctx) {
        close(ctx->epoll_fd);
        free(ctx);
    }
}

int io_context_add(io_context_t* ctx, socket_t fd, int events, void* user_data) {
    struct epoll_event ev;
    ev.events = 0;
    if (events & IO_EVENT_READ) ev.events |= EPOLLIN;
    if (events & IO_EVENT_WRITE) ev.events |= EPOLLOUT;
    ev.data.ptr = user_data;
    
    return epoll_ctl(ctx->epoll_fd, EPOLL_CTL_ADD, fd, &ev);
}

int io_context_remove(io_context_t* ctx, socket_t fd) {
    return epoll_ctl(ctx->epoll_fd, EPOLL_CTL_DEL, fd, NULL);
}

int io_context_wait(io_context_t* ctx, io_event_t* events, int max_events, int timeout_ms) {
    struct epoll_event epoll_events[max_events];
    int nfds = epoll_wait(ctx->epoll_fd, epoll_events, max_events, timeout_ms);
    
    for (int i = 0; i < nfds; i++) {
        events[i].fd = 0;  // Need to track fd separately
        events[i].revents = 0;
        if (epoll_events[i].events & EPOLLIN) events[i].revents |= IO_EVENT_READ;
        if (epoll_events[i].events & EPOLLOUT) events[i].revents |= IO_EVENT_WRITE;
        if (epoll_events[i].events & EPOLLERR) events[i].revents |= IO_EVENT_ERROR;
        events[i].user_data = epoll_events[i].data.ptr;
    }
    
    return nfds;
}

#else
// Fallback select() implementation for other platforms
#include <sys/select.h>

struct io_context {
    fd_set read_fds, write_fds;
    socket_t max_fd;
    int max_events;
};

io_context_t* io_context_create(int max_events) {
    io_context_t* ctx = malloc(sizeof(io_context_t));
    if (!ctx) return NULL;
    
    FD_ZERO(&ctx->read_fds);
    FD_ZERO(&ctx->write_fds);
    ctx->max_fd = 0;
    ctx->max_events = max_events;
    
    return ctx;
}

void io_context_destroy(io_context_t* ctx) {
    if (ctx) {
        free(ctx);
    }
}

int io_context_add(io_context_t* ctx, socket_t fd, int events, void* user_data) {
    if (events & IO_EVENT_READ) FD_SET(fd, &ctx->read_fds);
    if (events & IO_EVENT_WRITE) FD_SET(fd, &ctx->write_fds);
    if (fd > ctx->max_fd) ctx->max_fd = fd;
    return 0;
}

int io_context_remove(io_context_t* ctx, socket_t fd) {
    FD_CLR(fd, &ctx->read_fds);
    FD_CLR(fd, &ctx->write_fds);
    return 0;
}

int io_context_wait(io_context_t* ctx, io_event_t* events, int max_events, int timeout_ms) {
    fd_set read_fds = ctx->read_fds;
    fd_set write_fds = ctx->write_fds;
    
    struct timeval timeout;
    struct timeval* timeout_ptr = NULL;
    
    if (timeout_ms >= 0) {
        timeout.tv_sec = timeout_ms / 1000;
        timeout.tv_usec = (timeout_ms % 1000) * 1000;
        timeout_ptr = &timeout;
    }
    
    int result = select(ctx->max_fd + 1, &read_fds, &write_fds, NULL, timeout_ptr);
    if (result <= 0) return result;
    
    int event_count = 0;
    for (socket_t fd = 0; fd <= ctx->max_fd && event_count < max_events; fd++) {
        int revents = 0;
        if (FD_ISSET(fd, &read_fds)) revents |= IO_EVENT_READ;
        if (FD_ISSET(fd, &write_fds)) revents |= IO_EVENT_WRITE;
        
        if (revents) {
            events[event_count].fd = fd;
            events[event_count].revents = revents;
            events[event_count].user_data = NULL;  // Not supported in this implementation
            event_count++;
        }
    }
    
    return event_count;
}
#endif
```

### 5. Cross-platform Server Example

```c
// cross_platform_server.c - Complete cross-platform server
#include "platform.h"
#include "io_multiplex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CLIENTS 1000
#define BUFFER_SIZE 1024

typedef struct {
    socket_t fd;
    char buffer[BUFFER_SIZE];
    size_t buffer_len;
} client_info_t;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <port>\n", argv[0]);
        return 1;
    }
    
    int port = atoi(argv[1]);
    
    // Initialize socket library
    if (socket_library_init() < 0) {
        return 1;
    }
    
    // Create server socket
    socket_t server_fd = create_tcp_socket();
    if (server_fd == INVALID_SOCKET_FD) {
        socket_library_cleanup();
        return 1;
    }
    
    // Configure socket
    set_socket_reuse_addr(server_fd);
    set_socket_nonblocking(server_fd);
    
    // Bind and listen
    if (bind_socket(server_fd, NULL, port) < 0) {
        close_socket(server_fd);
        socket_library_cleanup();
        return 1;
    }
    
    if (listen(server_fd, 5) < 0) {
        fprintf(stderr, "Listen failed: %s\n", 
                socket_error_string(get_socket_error()));
        close_socket(server_fd);
        socket_library_cleanup();
        return 1;
    }
    
    printf("Server listening on port %d\n", port);
    
    // Create I/O context
    io_context_t* io_ctx = io_context_create(MAX_CLIENTS);
    if (!io_ctx) {
        fprintf(stderr, "Failed to create I/O context\n");
        close_socket(server_fd);
        socket_library_cleanup();
        return 1;
    }
    
    // Add server socket to I/O context
    io_context_add(io_ctx, server_fd, IO_EVENT_READ, NULL);
    
    // Client management
    client_info_t* clients[MAX_CLIENTS] = {0};
    
    // Main event loop
    io_event_t events[100];
    while (1) {
        int event_count = io_context_wait(io_ctx, events, 100, -1);
        
        for (int i = 0; i < event_count; i++) {
            if (events[i].fd == server_fd) {
                // New client connection
                struct sockaddr_in client_addr;
                socklen_t client_len = sizeof(client_addr);
                
                socket_t client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
                if (client_fd != INVALID_SOCKET_FD) {
                    set_socket_nonblocking(client_fd);
                    
                    // Find free client slot
                    for (int j = 0; j < MAX_CLIENTS; j++) {
                        if (!clients[j]) {
                            clients[j] = malloc(sizeof(client_info_t));
                            clients[j]->fd = client_fd;
                            clients[j]->buffer_len = 0;
                            
                            io_context_add(io_ctx, client_fd, IO_EVENT_READ, clients[j]);
                            printf("Client connected\n");
                            break;
                        }
                    }
                }
            } else {
                // Client data or disconnection
                client_info_t* client = (client_info_t*)events[i].user_data;
                if (!client) continue;
                
                if (events[i].revents & IO_EVENT_READ) {
                    char temp_buffer[BUFFER_SIZE];
                    ssize_t bytes_read = recv(client->fd, temp_buffer, sizeof(temp_buffer), 0);
                    
                    if (bytes_read <= 0) {
                        // Client disconnected
                        printf("Client disconnected\n");
                        io_context_remove(io_ctx, client->fd);
                        close_socket(client->fd);
                        
                        // Remove from client list
                        for (int j = 0; j < MAX_CLIENTS; j++) {
                            if (clients[j] == client) {
                                free(clients[j]);
                                clients[j] = NULL;
                                break;
                            }
                        }
                    } else {
                        // Echo data back
                        send(client->fd, temp_buffer, bytes_read, 0);
                    }
                }
            }
        }
    }
    
    // Cleanup
    io_context_destroy(io_ctx);
    close_socket(server_fd);
    socket_library_cleanup();
    
    return 0;
}
```

### 6. CMake Configuration

```cmake
# CMakeLists.txt - Cross-platform build configuration
cmake_minimum_required(VERSION 3.10)
project(CrossPlatformSocket)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Source files
set(SOURCES
    socket_init.c
    socket_ops.c
    io_multiplex.c
    cross_platform_server.c
)

# Platform-specific configurations
if(WIN32)
    # Windows-specific settings
    add_definitions(-DPLATFORM_WINDOWS)
    set(PLATFORM_LIBS ws2_32 mswsock)
    
    # Suppress Windows warnings
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_WINSOCK_DEPRECATED_NO_WARNINGS)
    
elseif(UNIX AND NOT APPLE)
    # Linux-specific settings
    add_definitions(-DPLATFORM_LINUX)
    set(PLATFORM_LIBS "")
    
    # Enable GNU extensions
    add_definitions(-D_GNU_SOURCE)
    
elseif(APPLE)
    # macOS-specific settings
    add_definitions(-DPLATFORM_MACOS)
    set(PLATFORM_LIBS "")
    
else()
    # Generic Unix
    add_definitions(-DPLATFORM_UNIX)
    set(PLATFORM_LIBS "")
endif()

# Create executable
add_executable(cross_platform_server ${SOURCES})

# Link libraries
target_link_libraries(cross_platform_server ${PLATFORM_LIBS})

# Compiler-specific flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cross_platform_server PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
    )
endif()

if(MSVC)
    target_compile_options(cross_platform_server PRIVATE
        /W4
        /wd4996  # Disable deprecated function warnings
    )
endif()

# Install target
install(TARGETS cross_platform_server
        RUNTIME DESTINATION bin)
```

## Platform Differences Summary

### Socket Types and Constants

| Aspect | Windows | POSIX |
|--------|---------|-------|
| Socket type | `SOCKET` (unsigned int) | `int` |
| Invalid socket | `INVALID_SOCKET` | `-1` |
| Socket error | `SOCKET_ERROR` | `-1` |
| Close function | `closesocket()` | `close()` |
| Error code | `WSAGetLastError()` | `errno` |

### Error Codes

| Error | Windows | POSIX |
|-------|---------|-------|
| Would block | `WSAEWOULDBLOCK` | `EAGAIN`/`EWOULDBLOCK` |
| In progress | `WSAEINPROGRESS` | `EINPROGRESS` |
| Connection refused | `WSAECONNREFUSED` | `ECONNREFUSED` |
| Address in use | `WSAEADDRINUSE` | `EADDRINUSE` |

### I/O Models

| Platform | Best I/O Model | Alternative |
|----------|----------------|-------------|
| Windows | IOCP | select() |
| Linux | epoll | select(), poll() |
| macOS/BSD | kqueue | select(), poll() |
| Generic Unix | select() | poll() |

## Best Practices

### Code Organization
1. **Separate platform-specific code** into different files
2. **Use feature detection** rather than platform detection when possible
3. **Create consistent interfaces** across platforms
4. **Test on all target platforms** regularly

### Error Handling
1. **Abstract error handling** to hide platform differences
2. **Provide consistent error codes** across platforms
3. **Include detailed error messages** for debugging
4. **Handle platform-specific edge cases**

### Performance Considerations
1. **Use platform-optimized I/O models** (IOCP, epoll, kqueue)
2. **Configure socket options** appropriately for each platform
3. **Profile on target platforms** to identify bottlenecks
4. **Consider platform-specific optimizations**

### Build System
1. **Use CMake** for cross-platform builds
2. **Detect features at build time** when possible
3. **Handle dependencies** cleanly across platforms
4. **Provide platform-specific build instructions**

## Assessment Checklist

- [ ] Understands differences between Winsock and POSIX sockets
- [ ] Can write portable socket code with proper abstractions
- [ ] Handles platform-specific considerations correctly
- [ ] Manages error codes and differences across platforms
- [ ] Configures cross-platform build systems effectively
- [ ] Tests applications on multiple platforms

## Next Steps

After mastering cross-platform socket programming:
- Explore platform-specific optimizations (IOCP, io_uring)
- Study high-performance cross-platform libraries (libuv, Boost.Asio)
- Learn about mobile platform considerations (iOS, Android)

## Resources

- "Network Programming for Microsoft Windows" by Anthony Jones and Jim Ohlund
- "UNIX Network Programming, Volume 1" by W. Richard Stevens
- Microsoft Winsock documentation
- POSIX socket specifications
- CMake documentation for cross-platform development

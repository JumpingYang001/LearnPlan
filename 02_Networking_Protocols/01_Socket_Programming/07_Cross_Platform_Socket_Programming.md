# Cross-Platform Socket Programming

*Last Updated: June 21, 2025*


## Overview

Cross-platform socket programming enables you to write network code that runs seamlessly on Windows, Linux, macOS, and other operating systems. This is essential for building portable, maintainable, and robust networked applications.

**Why is cross-platform socket programming important?**
- Most real-world software must run on more than one OS (e.g., servers, tools, games)
- Each OS has its own socket API quirks and best practices
- Writing portable code saves time and reduces bugs

**Key Challenges:**
- Different APIs (Winsock vs POSIX)
- Data type and constant differences
- Error handling and resource management
- Build system and compiler differences

**This module will teach you:**
- How to detect and abstract platform differences
- How to write code that compiles and runs everywhere
- How to debug and test cross-platform network code

**Visual: Cross-Platform Socket Architecture**
```
┌────────────┐   ┌────────────┐   ┌────────────┐
│  Windows   │   │   Linux    │   │   macOS    │
│  (Winsock) │   │  (POSIX)   │   │  (POSIX)   │
└─────┬──────┘   └────┬───────┘   └────┬───────┘
      │                │                │
      └─────┬──────────┴─────┬─────────┘
            │  Abstraction   │
            │  Layer         │
            └─────┬──────────┘
                  │
           ┌──────▼──────┐
           │ Your App    │
           └─────────────┘
```


## Learning Objectives

By the end of this module, you will be able to:
- **Explain** the differences between Windows Winsock and POSIX sockets (with examples)
- **Write** portable socket code using abstraction layers and macros
- **Handle** platform-specific issues (types, errors, initialization, cleanup)
- **Build** and test cross-platform network applications using CMake
- **Debug** and troubleshoot cross-platform socket issues

### Self-Assessment Checklist

Before moving on, make sure you can:
□ List key differences between Winsock and POSIX sockets  
□ Write a function that works on both Windows and Linux  
□ Set up a CMake build for a portable network app  
□ Debug a socket error on both platforms  
□ Explain how to handle non-blocking sockets everywhere

## Topics Covered


### 1. Windows Socket API (Winsock)
**Winsock** is the Windows API for network programming. It requires explicit initialization and cleanup, and uses different types and error codes than POSIX.

- **Initialization:** `WSAStartup()` must be called before any socket functions.
- **Cleanup:** `WSACleanup()` must be called before program exit.
- **Socket type:** `SOCKET` (not `int`)
- **Close function:** `closesocket()`
- **Error codes:** Use `WSAGetLastError()`
- **Advanced I/O:** IOCP (I/O Completion Ports) for high-performance servers

**Example:**
```c
WSADATA wsaData;
if (WSAStartup(MAKEWORD(2,2), &wsaData) != 0) {
    // Handle error
}
// ... use sockets ...
WSACleanup();
```

### 2. POSIX Socket API
**POSIX sockets** are used on Linux, macOS, BSD, and most Unix systems. They are based on the BSD sockets API.

- **No explicit initialization/cleanup**
- **Socket type:** `int`
- **Close function:** `close()`
- **Error codes:** Use `errno`
- **Advanced I/O:** `epoll` (Linux), `kqueue` (BSD/macOS)

**Example:**
```c
int sockfd = socket(AF_INET, SOCK_STREAM, 0);
if (sockfd < 0) {
    perror("socket");
}
// ... use sockets ...
close(sockfd);
```

### 3. Abstraction Layers
To write portable code, use macros and wrapper functions to hide platform differences.

- **Macros:** Detect platform and define types/constants
- **Wrapper functions:** Provide a unified API for your app
- **Configuration management:** Use CMake or similar tools to manage platform-specific code

**Example:**
```c
#ifdef _WIN32
    #define CLOSE_SOCKET(s) closesocket(s)
#else
    #define CLOSE_SOCKET(s) close(s)
#endif
```

### 4. Platform-specific Considerations
- **Error handling:** Different error codes and functions
- **Data types:** `SOCKET` vs `int`, `socklen_t` differences
- **Build system:** Compiler flags, library linking, header locations
- **I/O models:** IOCP, epoll, kqueue, select

### 5. Cross-platform Build Systems
- **CMake:** The de facto standard for C/C++ cross-platform builds
- **Conditional compilation:** Use `if(WIN32)` or `#ifdef _WIN32` in code and CMake
- **Dependency management:** Link correct libraries for each platform


## Practical Exercises

### 1. Platform Detection System
- **Goal:** Write macros to detect Windows, Linux, macOS, and use them to select code paths.
- **Exercise:**
    - Write a header file that defines `PLATFORM_WINDOWS`, `PLATFORM_LINUX`, `PLATFORM_MACOS` based on preprocessor macros.
    - Print the detected platform at runtime.

### 2. Portable Socket Library
- **Goal:** Build a C file with functions for `socket_create()`, `socket_close()`, etc., that work on all platforms.
- **Exercise:**
    - Implement wrappers for socket creation, closing, and error handling.
    - Test your code on at least two platforms (e.g., Windows and Linux).

### 3. Cross-platform Server
- **Goal:** Write a TCP echo server that compiles and runs on Windows, Linux, and macOS.
- **Exercise:**
    - Use your abstraction layer from Exercise 2.
    - Accept multiple clients and echo data back.
    - Use non-blocking sockets and I/O multiplexing (select/epoll/IOCP).

### 4. Build System Setup
- **Goal:** Use CMake to build your project on all platforms.
- **Exercise:**
    - Write a `CMakeLists.txt` that detects the platform and links the correct libraries.
    - Add compiler flags for warnings and debugging.

### 5. Debugging and Troubleshooting
- **Goal:** Learn to debug cross-platform socket issues.
- **Exercise:**
    - Intentionally introduce a bug (e.g., forget to call `WSAStartup()` on Windows).
    - Observe the error and fix it.

---

**Tip:** Try to run your code in a virtual machine or container for each OS if you don't have physical access.

## Code Examples


### 1. Platform Detection and Basic Setup

#### Explanation
To write portable code, you must detect the platform at compile time and use the correct headers, types, and functions. This is usually done with preprocessor macros.

**Diagram: Platform Detection Flow**
```
┌────────────┐
│  _WIN32?   │──Yes──> Windows code
└─────┬──────┘
      │
      No
      │
┌────────────┐
│ __linux__? │──Yes──> Linux code
└─────┬──────┘
      │
      ...
```

**Best Practice:**
- Always keep platform detection in a single header (e.g., `platform.h`)
- Use typedefs and macros to hide differences

**Code Walkthrough:**

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

#### Explanation
**Windows:** You must call `WSAStartup()` before using any socket functions, and `WSACleanup()` before exit. Forgetting this is a common source of bugs.

**POSIX:** No explicit initialization or cleanup is needed. Sockets are ready to use after including the right headers.

**Best Practice:**
- Always wrap initialization and cleanup in functions (e.g., `socket_library_init()`, `socket_library_cleanup()`).
- Call these at the start and end of your program.

**Troubleshooting:**
- If you get `WSANOTINITIALISED` on Windows, you forgot `WSAStartup()`.
- On POSIX, check for missing headers or permissions.

**Code Walkthrough:**

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

#### Explanation
Portable socket operations mean writing code that works the same way on all platforms. This includes creating, binding, connecting, and closing sockets, as well as setting options like non-blocking mode.

**Key Differences:**
- Windows uses `SOCKET` type, POSIX uses `int`
- Non-blocking mode: `ioctlsocket()` (Windows) vs `fcntl()` (POSIX)
- Closing: `closesocket()` (Windows) vs `close()` (POSIX)

**Best Practice:**
- Use wrapper functions for all socket operations
- Always check return values and print error messages using your abstraction

**Code Walkthrough:**

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

#### Explanation
I/O multiplexing lets you handle many sockets at once without blocking. Each platform has its own best model:
- **Windows:** IOCP (I/O Completion Ports)
- **Linux:** epoll
- **macOS/BSD:** kqueue
- **Fallback:** select()

**Diagram: I/O Multiplexing Models**
```
Windows ──> IOCP
Linux   ──> epoll
macOS   ──> kqueue
Other   ──> select()
```

**Best Practice:**
- Use the best model for each platform, but provide a fallback (select) for portability
- Abstract the I/O context and event loop

**Code Walkthrough:**

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

#### Explanation
This is a complete TCP echo server that works on Windows, Linux, and macOS. It uses all the abstractions and techniques described above.

**How it works:**
1. Initializes the socket library (if needed)
2. Creates a listening socket
3. Sets options (reuse address, non-blocking)
4. Binds and listens on a port
5. Uses the best I/O multiplexing model for the platform
6. Accepts clients and echoes data back
7. Cleans up all resources on exit

**Best Practice:**
- Always check for errors at every step
- Use non-blocking sockets for scalability
- Clean up all sockets and memory before exit

**Code Walkthrough:**

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

#### Explanation
CMake is the standard tool for building C/C++ projects on all platforms. It lets you write one build script that works everywhere.

**Key Points:**
- Use `if(WIN32)`, `if(UNIX)`, `if(APPLE)` to detect platforms
- Link the correct libraries for each OS
- Add warning and debug flags for all compilers

**Best Practice:**
- Keep your CMakeLists.txt simple and well-commented
- Test your build on all target platforms

**Code Walkthrough:**

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

#### Visual Table: Key Differences

| Aspect         | Windows (Winsock)      | POSIX (Linux/macOS/BSD) |
|--------------- |-----------------------|-------------------------|
| Socket type    | `SOCKET` (unsigned)   | `int`                   |
| Invalid socket | `INVALID_SOCKET`      | `-1`                    |
| Error code     | `WSAGetLastError()`   | `errno`                 |
| Close socket   | `closesocket()`       | `close()`               |
| Init/cleanup   | `WSAStartup()`/`WSACleanup()` | None         |
| I/O model      | IOCP, select()        | epoll, kqueue, select() |

**Tip:** Always use your abstraction macros and functions to hide these differences!

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
1. **Separate platform-specific code** into different files or use `#ifdef` blocks
2. **Keep platform detection in one header** (e.g., `platform.h`)
3. **Use wrapper functions** for all socket operations
4. **Test on all platforms** early and often

### Error Handling
1. **Abstract error handling** so your app doesn't care about platform details
2. **Print detailed error messages** for easier debugging
3. **Handle all error cases** (not just the common ones)

### Performance
1. **Use the best I/O model** for each platform (IOCP, epoll, kqueue)
2. **Profile and tune** on each OS

### Build System
1. **Use CMake** for all builds
2. **Document platform-specific steps** in your README

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

**Self-Assessment:**

□ Can you explain the main differences between Winsock and POSIX sockets?  
□ Can you write a function that works on both Windows and Linux?  
□ Can you set up a CMake build for a portable network app?  
□ Can you debug a socket error on both platforms?  
□ Can you handle non-blocking sockets everywhere?  
□ Have you tested your code on at least two platforms?  

## Next Steps

After mastering cross-platform socket programming:
- Explore platform-specific optimizations (IOCP, io_uring)
- Study high-performance cross-platform libraries (libuv, Boost.Asio)
- Learn about mobile platform considerations (iOS, Android)


## Resources

- "Network Programming for Microsoft Windows" by Anthony Jones and Jim Ohlund
- "UNIX Network Programming, Volume 1" by W. Richard Stevens
- Microsoft Winsock documentation ([MSDN](https://docs.microsoft.com/en-us/windows/win32/winsock/))
- POSIX socket specifications ([The Open Group](https://pubs.opengroup.org/onlinepubs/9699919799/functions/socket.html))
- CMake documentation for cross-platform development ([cmake.org](https://cmake.org/cmake/help/latest/))
- [Beej's Guide to Network Programming](https://beej.us/guide/bgnet/)
- [LLNL Pthreads and Sockets Tutorial](https://computing.llnl.gov/tutorials/pthreads/)

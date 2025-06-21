# Socket I/O Models

*Last Updated: June 21, 2025*

## Overview

This module covers different I/O models for socket programming, from simple blocking I/O to advanced event-driven architectures. Understanding these models is crucial for building scalable and high-performance network applications.

## Learning Objectives

By the end of this module, you should be able to:
- Understand different I/O models and their trade-offs
- Implement blocking and non-blocking I/O patterns
- Use I/O multiplexing with select() and poll()
- Design event-driven applications with epoll/kqueue/IOCP
- Choose appropriate I/O models for different scenarios

## Topics Covered

### 1. Blocking I/O
- Synchronous communication patterns
- Timeout handling mechanisms
- Advantages and limitations

### 2. Non-blocking I/O
- Setting non-blocking mode
- Polling with non-blocking sockets
- Handling EAGAIN/EWOULDBLOCK

### 3. I/O Multiplexing
- select() system call
- poll() system call
- Implementation patterns and best practices

### 4. Event-driven I/O
- epoll (Linux) - Edge-triggered and Level-triggered
- kqueue (BSD/macOS)
- IOCP (Windows)
- Event notification mechanisms

## Practical Exercises

1. **Blocking vs Non-blocking Comparison**
   - Implement same functionality with both models
   - Measure performance differences

2. **Select-based Server**
   - Multi-client server using select()
   - Handle both TCP and UDP sockets

3. **Epoll-based High-Performance Server**
   - Event-driven server architecture
   - Handle thousands of concurrent connections

4. **Cross-platform I/O Abstraction**
   - Unified interface for different platforms
   - Performance comparison across platforms

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

| Model | Scalability | Complexity | CPU Usage | Memory Usage | Best For |
|-------|-------------|------------|-----------|--------------|----------|
| Blocking | Low | Low | High | Low | Simple applications |
| Non-blocking | Medium | Medium | Medium | Low | Interactive applications |
| Select/Poll | Medium | Medium | Medium | Medium | Moderate load |
| Epoll/Kqueue | High | High | Low | Medium | High load servers |
| IOCP | High | High | Low | Medium | Windows servers |

## Performance Considerations

### When to Use Each Model

1. **Blocking I/O**: Simple applications, low concurrency requirements
2. **Non-blocking I/O**: Applications needing responsiveness during I/O
3. **Select/Poll**: Moderate number of connections (< 1000)
4. **Epoll/Kqueue**: High-performance servers (> 1000 connections)
5. **IOCP**: Windows-based high-performance applications

### Optimization Tips

1. **Use appropriate buffer sizes** for your workload
2. **Minimize system calls** by batching operations
3. **Consider edge-triggered vs level-triggered** modes
4. **Profile your application** to identify bottlenecks
5. **Test under realistic load** conditions

## Assessment Checklist

- [ ] Understands different I/O models and their trade-offs
- [ ] Can implement blocking I/O with timeouts
- [ ] Successfully uses non-blocking I/O patterns
- [ ] Implements I/O multiplexing with select/poll
- [ ] Creates event-driven applications with epoll/kqueue
- [ ] Chooses appropriate I/O models for different scenarios

## Next Steps

After mastering I/O models:
- Explore async/await patterns in modern languages
- Study high-performance networking libraries (libuv, Boost.Asio)
- Learn about IOCP for Windows development

## Resources

- "UNIX Network Programming, Volume 1" by W. Richard Stevens (Chapter 6)
- "The C10K Problem" by Dan Kegel
- Linux man pages: select(2), poll(2), epoll(7)
- "Scalable Network Programming" articles and tutorials

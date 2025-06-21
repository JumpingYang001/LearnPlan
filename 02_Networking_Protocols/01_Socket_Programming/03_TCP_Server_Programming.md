# TCP Socket Programming - Server Side

*Last Updated: June 21, 2025*

## Overview

This module covers TCP server implementation using socket programming. You'll learn how to create servers that can accept multiple client connections, handle concurrent clients, and implement various server architecture patterns.

## Learning Objectives

By the end of this module, you should be able to:
- Create and configure TCP server sockets
- Bind sockets to addresses and listen for connections
- Accept and handle client connections
- Implement concurrent server designs
- Choose appropriate server scaling patterns

## Topics Covered

### Socket Creation and Binding
- Server socket creation
- Address binding with `bind()`
- Handling address reuse (SO_REUSEADDR)
- Port binding considerations

### Listening for Connections
- `listen()` system call and backlog queue
- Connection queue management
- Backlog size considerations

### Accepting Client Connections
- `accept()` system call
- Client address information
- Connection handling patterns

### Concurrent Server Designs
- Iterative servers (one client at a time)
- Process-per-client servers (`fork()`)
- Thread-per-client servers (`pthread_create()`)
- Thread pool servers
- Event-driven servers

### Server Scaling Patterns
- Connection limits and resource management
- Load balancing considerations
- Performance optimization techniques

## Practical Exercises

1. **Simple Echo Server**
   - Implement a basic iterative echo server
   - Handle one client at a time

2. **Multi-threaded Echo Server**
   - Handle multiple clients concurrently
   - Use pthread for thread management

3. **Chat Server**
   - Broadcast messages to all connected clients
   - Maintain client list and handle disconnections

4. **File Server**
   - Serve files to clients
   - Handle multiple file requests simultaneously

## Code Examples

### Basic TCP Server Setup
```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>

int create_tcp_server(int port) {
    int server_fd;
    struct sockaddr_in server_addr;
    int opt = 1;
    
    // Create socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("Socket creation failed");
        return -1;
    }
    
    // Set socket options
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("Setsockopt failed");
        close(server_fd);
        return -1;
    }
    
    // Configure server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);
    
    // Bind socket
    if (bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Bind failed");
        close(server_fd);
        return -1;
    }
    
    // Start listening
    if (listen(server_fd, 5) < 0) {
        perror("Listen failed");
        close(server_fd);
        return -1;
    }
    
    printf("Server listening on port %d\n", port);
    return server_fd;
}
```

### Iterative Server
```c
void iterative_server(int server_fd) {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int client_fd;
    char buffer[1024];
    
    while (1) {
        // Accept client connection
        client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            perror("Accept failed");
            continue;
        }
        
        printf("Client connected: %s:%d\n", 
               inet_ntoa(client_addr.sin_addr), 
               ntohs(client_addr.sin_port));
        
        // Handle client (echo server)
        ssize_t bytes_received;
        while ((bytes_received = recv(client_fd, buffer, sizeof(buffer), 0)) > 0) {
            if (send(client_fd, buffer, bytes_received, 0) < 0) {
                perror("Send failed");
                break;
            }
        }
        
        printf("Client disconnected\n");
        close(client_fd);
    }
}
```

### Multi-threaded Server
```c
#include <pthread.h>

typedef struct {
    int client_fd;
    struct sockaddr_in client_addr;
} client_info_t;

void* handle_client(void* arg) {
    client_info_t* client = (client_info_t*)arg;
    char buffer[1024];
    ssize_t bytes_received;
    
    printf("Thread handling client: %s:%d\n", 
           inet_ntoa(client->client_addr.sin_addr), 
           ntohs(client->client_addr.sin_port));
    
    // Handle client communication
    while ((bytes_received = recv(client->client_fd, buffer, sizeof(buffer), 0)) > 0) {
        if (send(client->client_fd, buffer, bytes_received, 0) < 0) {
            perror("Send failed");
            break;
        }
    }
    
    printf("Client thread terminating\n");
    close(client->client_fd);
    free(client);
    return NULL;
}

void multithreaded_server(int server_fd) {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int client_fd;
    pthread_t thread_id;
    
    while (1) {
        client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            perror("Accept failed");
            continue;
        }
        
        // Allocate client info
        client_info_t* client = malloc(sizeof(client_info_t));
        client->client_fd = client_fd;
        client->client_addr = client_addr;
        
        // Create thread to handle client
        if (pthread_create(&thread_id, NULL, handle_client, client) != 0) {
            perror("Thread creation failed");
            close(client_fd);
            free(client);
            continue;
        }
        
        // Detach thread so it cleans up automatically
        pthread_detach(thread_id);
    }
}
```

### Thread Pool Server
```c
#include <pthread.h>
#include <semaphore.h>

#define THREAD_POOL_SIZE 10
#define QUEUE_SIZE 100

typedef struct {
    int client_fds[QUEUE_SIZE];
    int front, rear, count;
    pthread_mutex_t mutex;
    sem_t empty_slots;
    sem_t filled_slots;
} client_queue_t;

client_queue_t client_queue = {
    .front = 0, .rear = 0, .count = 0,
    .mutex = PTHREAD_MUTEX_INITIALIZER
};

void queue_init() {
    sem_init(&client_queue.empty_slots, 0, QUEUE_SIZE);
    sem_init(&client_queue.filled_slots, 0, 0);
}

void enqueue_client(int client_fd) {
    sem_wait(&client_queue.empty_slots);
    pthread_mutex_lock(&client_queue.mutex);
    
    client_queue.client_fds[client_queue.rear] = client_fd;
    client_queue.rear = (client_queue.rear + 1) % QUEUE_SIZE;
    client_queue.count++;
    
    pthread_mutex_unlock(&client_queue.mutex);
    sem_post(&client_queue.filled_slots);
}

int dequeue_client() {
    sem_wait(&client_queue.filled_slots);
    pthread_mutex_lock(&client_queue.mutex);
    
    int client_fd = client_queue.client_fds[client_queue.front];
    client_queue.front = (client_queue.front + 1) % QUEUE_SIZE;
    client_queue.count--;
    
    pthread_mutex_unlock(&client_queue.mutex);
    sem_post(&client_queue.empty_slots);
    
    return client_fd;
}

void* worker_thread(void* arg) {
    char buffer[1024];
    
    while (1) {
        int client_fd = dequeue_client();
        
        // Handle client
        ssize_t bytes_received;
        while ((bytes_received = recv(client_fd, buffer, sizeof(buffer), 0)) > 0) {
            if (send(client_fd, buffer, bytes_received, 0) < 0) {
                break;
            }
        }
        
        close(client_fd);
    }
    
    return NULL;
}

void thread_pool_server(int server_fd) {
    pthread_t workers[THREAD_POOL_SIZE];
    
    queue_init();
    
    // Create worker threads
    for (int i = 0; i < THREAD_POOL_SIZE; i++) {
        pthread_create(&workers[i], NULL, worker_thread, NULL);
    }
    
    // Accept connections and add to queue
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    
    while (1) {
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            perror("Accept failed");
            continue;
        }
        
        enqueue_client(client_fd);
    }
}
```

### Process-per-Client Server
```c
#include <sys/wait.h>
#include <signal.h>

void sigchld_handler(int sig) {
    // Clean up zombie processes
    while (waitpid(-1, NULL, WNOHANG) > 0);
}

void process_per_client_server(int server_fd) {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    pid_t child_pid;
    
    // Set up signal handler for child processes
    signal(SIGCHLD, sigchld_handler);
    
    while (1) {
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            perror("Accept failed");
            continue;
        }
        
        child_pid = fork();
        if (child_pid == 0) {
            // Child process
            close(server_fd);  // Child doesn't need server socket
            
            // Handle client
            char buffer[1024];
            ssize_t bytes_received;
            while ((bytes_received = recv(client_fd, buffer, sizeof(buffer), 0)) > 0) {
                if (send(client_fd, buffer, bytes_received, 0) < 0) {
                    break;
                }
            }
            
            close(client_fd);
            exit(0);
        } else if (child_pid > 0) {
            // Parent process
            close(client_fd);  // Parent doesn't need client socket
        } else {
            perror("Fork failed");
            close(client_fd);
        }
    }
}
```

## Server Architecture Comparison

| Architecture | Pros | Cons | Best For |
|--------------|------|------|----------|
| Iterative | Simple, low resource usage | Only one client at a time | Testing, simple protocols |
| Process-per-client | Isolation, fault tolerance | High memory usage, slow | CPU-intensive tasks |
| Thread-per-client | Lower overhead than processes | Threading complexity | I/O bound tasks |
| Thread pool | Controlled resource usage | Queue management complexity | High load scenarios |
| Event-driven | Very scalable | Complex programming model | High concurrency |

## Error Handling and Best Practices

1. **Always check return values** from system calls
2. **Handle SIGPIPE** when clients disconnect unexpectedly
3. **Set SO_REUSEADDR** to avoid "Address already in use" errors
4. **Implement proper cleanup** for threads and processes
5. **Monitor resource usage** (file descriptors, memory)
6. **Add logging** for debugging and monitoring

## Assessment Checklist

- [ ] Can create and configure TCP server sockets
- [ ] Successfully binds to addresses and listens for connections
- [ ] Accepts and handles client connections properly
- [ ] Implements concurrent server architectures
- [ ] Handles server errors and edge cases
- [ ] Chooses appropriate scaling patterns for different scenarios

## Next Steps

After mastering TCP server programming:
- Explore event-driven I/O models (epoll, kqueue, IOCP)
- Learn about load balancing and high-availability patterns
- Study advanced server optimization techniques

## Resources

- "UNIX Network Programming, Volume 1" by W. Richard Stevens (Chapters 4-6)
- [Beej's Guide - Simple Server Example](https://beej.us/guide/bgnet/html/#a-simple-stream-server)
- Linux man pages: bind(2), listen(2), accept(2)
- "The C10K Problem" by Dan Kegel

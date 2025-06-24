# Socket Programming Patterns

*Last Updated: June 21, 2025*

## Overview

This module covers common socket programming patterns and architectural designs for building robust, scalable network applications. You'll learn proven patterns for client-server communication, server architectures, and specialized networking scenarios.

---

## What Are Socket Programming Patterns?

Socket programming patterns are reusable solutions to common problems in network application design. They help you:
- Build reliable and scalable clients and servers
- Handle failures and network unreliability
- Structure code for maintainability and performance

**Why use patterns?**
- Avoid reinventing the wheel
- Learn from industry best practices
- Make your code easier to understand and extend

---

## Visual Overview

```
┌────────────┐      ┌────────────┐      ┌────────────┐
│  Client    │<--->│  Proxy     │<--->│  Server    │
└────────────┘      └────────────┘      └────────────┘
      │                  │                  │
      ▼                  ▼                  ▼
  [Client Patterns]  [Proxy Patterns]  [Server Patterns]
```

---

## Learning Objectives

By the end of this module, you should be able to:
- Implement common client-side patterns for reliable communication
- Design and implement various server architectures
- Create proxy and gateway patterns for network intermediation
- Choose appropriate patterns for different application requirements
- Scale applications using proven architectural patterns

## Topics Covered


### 1. Client Patterns

Client-side patterns help you build robust, efficient, and reliable network clients. These patterns address common challenges such as connection loss, resource management, and keeping connections alive.

#### Key Concepts
- **Reconnection Strategies:** How to recover from lost connections automatically.
- **Connection Pooling:** Efficiently reuse connections for high-throughput clients.
- **Heartbeat Mechanisms:** Detect dead connections and maintain liveness.
- **Request-Response Correlation:** Match responses to requests in asynchronous or pipelined protocols.

#### When to Use
- Building clients for unreliable networks
- High-performance or high-frequency request scenarios
- Long-lived connections (e.g., chat, games, trading)

#### Visual: Client Resilience
```
┌────────────┐   lost   ┌────────────┐
│  Client    │────────►│  Server    │
└────────────┘         └────────────┘
     ▲   │                ▲
     │   └─reconnect──────┘
```

---

### 2. Server Patterns
- Iterative servers for simple cases
- Concurrent servers using processes and threads
- Thread pool servers for controlled concurrency
- Event-driven servers for high scalability
- Hybrid architectures

### 3. Proxy and Gateway Patterns
- Forward and reverse proxy implementations
- Protocol translation gateways
- Load balancing proxies
- Caching proxies

### 4. Reliability Patterns
- Circuit breaker pattern
- Retry mechanisms with backoff
- Timeout handling strategies
- Graceful degradation

## Practical Exercises

1. **Resilient Client Implementation**
   - Implement reconnection logic with exponential backoff
   - Add connection pooling for multiple concurrent requests

2. **Multi-architecture Server Comparison**
   - Implement the same server using different patterns
   - Benchmark performance and resource usage

3. **HTTP Proxy Server**
   - Build a simple HTTP proxy with caching
   - Add load balancing capabilities

4. **Message Queue System**
   - Implement a simple message broker
   - Add reliability and persistence features

## Code Examples

### 1. Client Patterns

#### Reconnection with Exponential Backoff
```c
#include <time.h>
#include <math.h>

// ---

/**
 * Reconnection with Exponential Backoff
 *
 * When a client loses connection to the server, it should not retry immediately in a tight loop.
 * Exponential backoff increases the delay between retries, reducing server overload and network congestion.
 *
 * - Start with a small delay (e.g., 100ms)
 * - Double the delay after each failure, up to a maximum (e.g., 30s)
 * - Add random jitter to avoid "thundering herd" effect
 *
 * This pattern is used in real-world systems like HTTP clients, database drivers, and cloud SDKs.
 *
 * Visual:
 *
 *   Attempt 1   Attempt 2   Attempt 3   ...
 *   |-----|--------|-------------|........
 *
 * Best Practice: Always log failures and give up after a maximum number of retries.
 */


typedef struct {
    socket_t sockfd;
    char server_ip[16];
    int server_port;
    int max_retries;
    int base_delay_ms;
    int max_delay_ms;
    int current_retry;
    time_t last_attempt;
    int connected;
} resilient_client_t;

resilient_client_t* create_resilient_client(const char* ip, int port) {
    resilient_client_t* client = malloc(sizeof(resilient_client_t));
    if (!client) return NULL;
    
    client->sockfd = INVALID_SOCKET_FD;
    strncpy(client->server_ip, ip, sizeof(client->server_ip) - 1);
    client->server_port = port;
    client->max_retries = 10;
    client->base_delay_ms = 100;
    client->max_delay_ms = 30000;  // 30 seconds
    client->current_retry = 0;
    client->last_attempt = 0;
    client->connected = 0;
    
    return client;
}

int calculate_backoff_delay(resilient_client_t* client) {
    // Exponential backoff with jitter
    int delay = client->base_delay_ms * (1 << client->current_retry);
    if (delay > client->max_delay_ms) {
        delay = client->max_delay_ms;
    }
    
    // Add jitter (±25%)
    int jitter = delay / 4;
    delay += (rand() % (2 * jitter)) - jitter;
    
    return delay;
}

int resilient_connect(resilient_client_t* client) {
    if (client->connected) {
        return 0;  // Already connected
    }
    
    time_t now = time(NULL);
    
    // Check if we should delay before retry
    if (client->current_retry > 0) {
        int delay_ms = calculate_backoff_delay(client);
        if ((now - client->last_attempt) < (delay_ms / 1000)) {
            return -1;  // Still in backoff period
        }
    }
    
    // Attempt connection
    if (client->sockfd != INVALID_SOCKET_FD) {
        close_socket(client->sockfd);
    }
    
    client->sockfd = create_tcp_socket();
    if (client->sockfd == INVALID_SOCKET_FD) {
        return -1;
    }
    
    set_socket_nonblocking(client->sockfd);
    
    int result = connect_socket(client->sockfd, client->server_ip, client->server_port);
    client->last_attempt = now;
    
    if (result == 0) {
        // Connected immediately
        client->connected = 1;
        client->current_retry = 0;
        printf("Connected to %s:%d\n", client->server_ip, client->server_port);
        return 0;
    } else if (result == 1) {
        // Connection in progress (non-blocking)
        // Should check with select/poll later
        return 1;
    } else {
        // Connection failed
        client->current_retry++;
        if (client->current_retry >= client->max_retries) {
            printf("Max connection retries exceeded\n");
            return -2;
        }
        
        printf("Connection failed, retry %d/%d in %d ms\n", 
               client->current_retry, client->max_retries, 
               calculate_backoff_delay(client));
        
        close_socket(client->sockfd);
        client->sockfd = INVALID_SOCKET_FD;
        return -1;
    }
}

int resilient_send(resilient_client_t* client, const void* data, size_t len) {
    if (!client->connected) {
        if (resilient_connect(client) != 0) {
            return -1;
        }
    }
    
    ssize_t sent = send(client->sockfd, data, len, 0);
    if (sent < 0) {
        int error = get_socket_error();
#ifdef PLATFORM_WINDOWS
        if (error == WSAECONNRESET || error == WSAECONNABORTED) {
#else
        if (error == ECONNRESET || error == EPIPE) {
#endif
            // Connection lost, mark as disconnected
            client->connected = 0;
            close_socket(client->sockfd);
            client->sockfd = INVALID_SOCKET_FD;
            return -1;
        }
    }
    
    return sent;
}
```

#### Connection Pool Pattern
```c
#define POOL_SIZE 10

// ---

/**
 * Connection Pool Pattern
 *
 * For clients that make many short-lived or concurrent requests, creating a new connection for each request is inefficient.
 * A connection pool maintains a set of open connections that can be reused, reducing latency and resource usage.
 *
 * - Acquire a connection from the pool before sending a request
 * - Release it back to the pool after use
 * - Idle connections are closed after a timeout
 *
 * Used in: Database clients, HTTP clients, microservices
 *
 * Visual:
 *
 *   ┌────────────┐
 *   │ Connection │
 *   │   Pool     │
 *   └─────┬──────┘
 *         │
 *   ┌─────┴─────┐
 *   │ Client(s) │
 *   └───────────┘
 *
 * Best Practice: Limit pool size to avoid exhausting server resources.
 */

typedef struct {
    socket_t sockfd;
    int in_use;
    time_t last_used;
    char server_ip[16];
    int server_port;
} pooled_connection_t;

typedef struct {
    pooled_connection_t connections[POOL_SIZE];
    int pool_size;
    int timeout_seconds;
    pthread_mutex_t mutex;
} connection_pool_t;

connection_pool_t* create_connection_pool(const char* ip, int port, int timeout) {
    connection_pool_t* pool = malloc(sizeof(connection_pool_t));
    if (!pool) return NULL;
    
    pool->pool_size = POOL_SIZE;
    pool->timeout_seconds = timeout;
    pthread_mutex_init(&pool->mutex, NULL);
    
    for (int i = 0; i < POOL_SIZE; i++) {
        pool->connections[i].sockfd = INVALID_SOCKET_FD;
        pool->connections[i].in_use = 0;
        pool->connections[i].last_used = 0;
        strncpy(pool->connections[i].server_ip, ip, sizeof(pool->connections[i].server_ip) - 1);
        pool->connections[i].server_port = port;
    }
    
    return pool;
}

socket_t acquire_connection(connection_pool_t* pool) {
    pthread_mutex_lock(&pool->mutex);
    
    time_t now = time(NULL);
    
    // First, look for an existing idle connection
    for (int i = 0; i < pool->pool_size; i++) {
        pooled_connection_t* conn = &pool->connections[i];
        
        if (!conn->in_use && conn->sockfd != INVALID_SOCKET_FD) {
            // Check if connection is still valid
            if ((now - conn->last_used) > pool->timeout_seconds) {
                // Connection too old, close it
                close_socket(conn->sockfd);
                conn->sockfd = INVALID_SOCKET_FD;
            } else {
                // Reuse existing connection
                conn->in_use = 1;
                conn->last_used = now;
                pthread_mutex_unlock(&pool->mutex);
                return conn->sockfd;
            }
        }
    }
    
    // No reusable connection found, create new one
    for (int i = 0; i < pool->pool_size; i++) {
        pooled_connection_t* conn = &pool->connections[i];
        
        if (!conn->in_use && conn->sockfd == INVALID_SOCKET_FD) {
            socket_t sockfd = create_tcp_socket();
            if (sockfd != INVALID_SOCKET_FD) {
                if (connect_socket(sockfd, conn->server_ip, conn->server_port) == 0) {
                    conn->sockfd = sockfd;
                    conn->in_use = 1;
                    conn->last_used = now;
                    pthread_mutex_unlock(&pool->mutex);
                    return sockfd;
                } else {
                    close_socket(sockfd);
                }
            }
            break;
        }
    }
    
    pthread_mutex_unlock(&pool->mutex);
    return INVALID_SOCKET_FD;  // No connection available
}

void release_connection(connection_pool_t* pool, socket_t sockfd) {
    pthread_mutex_lock(&pool->mutex);
    
    for (int i = 0; i < pool->pool_size; i++) {
        if (pool->connections[i].sockfd == sockfd) {
            pool->connections[i].in_use = 0;
            pool->connections[i].last_used = time(NULL);
            break;
        }
    }
    
    pthread_mutex_unlock(&pool->mutex);
}
```

#### Heartbeat Pattern
```c
typedef struct {
    socket_t sockfd;
// ---

/**
 * Heartbeat Pattern
 *
 * In long-lived connections, network failures may not be detected immediately.
 * Heartbeats are small periodic messages sent to check if the connection is still alive.
 *
 * - Client sends "HEARTBEAT" every N seconds
 * - Server replies with "HEARTBEAT_ACK"
 * - If no ACK is received within a timeout, the connection is considered dead
 *
 * Used in: Messaging systems, trading platforms, multiplayer games
 *
 * Visual:
 *
 *   [Client]──HEARTBEAT──►[Server]
 *   [Client]◄─ACK─────────[Server]
 *
 * Best Practice: Tune heartbeat interval and timeout for your application's needs.
 */
    time_t last_heartbeat_sent;
    time_t last_heartbeat_received;
    int heartbeat_interval;
    int heartbeat_timeout;
    int connection_alive;
} heartbeat_client_t;

int send_heartbeat(heartbeat_client_t* client) {
    const char* heartbeat_msg = "HEARTBEAT";
    ssize_t sent = send(client->sockfd, heartbeat_msg, strlen(heartbeat_msg), 0);
    
    if (sent > 0) {
        client->last_heartbeat_sent = time(NULL);
        return 0;
    }
    
    return -1;
}

int process_heartbeat_response(heartbeat_client_t* client, const char* data) {
    if (strncmp(data, "HEARTBEAT_ACK", 13) == 0) {
        client->last_heartbeat_received = time(NULL);
        client->connection_alive = 1;
        return 1;
    }
    return 0;
}

void check_heartbeat_status(heartbeat_client_t* client) {
    time_t now = time(NULL);
    
    // Send heartbeat if needed
    if ((now - client->last_heartbeat_sent) >= client->heartbeat_interval) {
        send_heartbeat(client);
    }
    
    // Check if connection is alive
    if ((now - client->last_heartbeat_received) > client->heartbeat_timeout) {
        client->connection_alive = 0;
        printf("Connection appears to be dead - no heartbeat response\n");
    }
}
```

### 2. Server Patterns

Server-side patterns help you design scalable, efficient, and robust servers. The right pattern depends on your application's concurrency, performance, and resource requirements.

#### Key Concepts
- **Iterative Server:** Handles one client at a time (simple, not scalable)
- **Thread-per-Client:** Spawns a new thread for each client (easy, but can exhaust resources)
- **Thread Pool:** Uses a fixed pool of worker threads to handle many clients (efficient, scalable)
- **Event-driven:** Uses non-blocking I/O and event loops (best for very high concurrency)
- **Hybrid:** Combines patterns for complex needs

#### When to Use
- High-load servers (web, chat, game, proxy)
- When you need to control resource usage
- When you want to avoid the overhead of creating/destroying threads for each client

#### Visual: Thread Pool Server
```
┌────────────┐   accept   ┌────────────┐
│  Clients   │──────────►│  Work Queue│
└────────────┘           └─────┬──────┘
                                   │
                        ┌──────────┴──────────┐
                        │ Worker Threads Pool │
                        └─────────────────────┘
```

---

#### Thread Pool Server Pattern
```c
#include <pthread.h>
#include <semaphore.h>

// ---

/**
 * Thread Pool Server Pattern
 *
 * Instead of creating a new thread for every client (which can exhaust system resources),
 * a thread pool server maintains a fixed number of worker threads. Incoming client requests
 * are placed in a work queue, and worker threads pick up tasks as they become available.
 *
 * - Efficient for high-load servers
 * - Controls concurrency and resource usage
 * - Reduces thread creation/destruction overhead
 *
 * Used in: Web servers, chat servers, proxies, databases
 *
 * Visual:
 *
 *   [Client]───►[Work Queue]───►[Worker Thread]
 *
 * Best Practice: Tune the pool size based on CPU cores and expected load.
 */

#define MAX_THREADS 20
#define QUEUE_SIZE 100

typedef struct {
    socket_t client_fd;
    struct sockaddr_in client_addr;
} work_item_t;

typedef struct {
    work_item_t queue[QUEUE_SIZE];
    int head, tail, count;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    int shutdown;
} work_queue_t;

typedef struct {
    pthread_t threads[MAX_THREADS];
    int thread_count;
    work_queue_t work_queue;
    int running;
} thread_pool_t;

void work_queue_init(work_queue_t* queue) {
    queue->head = queue->tail = queue->count = 0;
    queue->shutdown = 0;
    pthread_mutex_init(&queue->mutex, NULL);
    pthread_cond_init(&queue->not_empty, NULL);
    pthread_cond_init(&queue->not_full, NULL);
}

void work_queue_destroy(work_queue_t* queue) {
    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->not_empty);
    pthread_cond_destroy(&queue->not_full);
}

int work_queue_push(work_queue_t* queue, work_item_t item) {
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->count == QUEUE_SIZE && !queue->shutdown) {
        pthread_cond_wait(&queue->not_full, &queue->mutex);
    }
    
    if (queue->shutdown) {
        pthread_mutex_unlock(&queue->mutex);
        return -1;
    }
    
    queue->queue[queue->tail] = item;
    queue->tail = (queue->tail + 1) % QUEUE_SIZE;
    queue->count++;
    
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);
    
    return 0;
}

int work_queue_pop(work_queue_t* queue, work_item_t* item) {
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->count == 0 && !queue->shutdown) {
        pthread_cond_wait(&queue->not_empty, &queue->mutex);
    }
    
    if (queue->shutdown && queue->count == 0) {
        pthread_mutex_unlock(&queue->mutex);
        return -1;
    }
    
    *item = queue->queue[queue->head];
    queue->head = (queue->head + 1) % QUEUE_SIZE;
    queue->count--;
    
    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);
    
    return 0;
}

void handle_client_request(socket_t client_fd, struct sockaddr_in* client_addr) {
    char buffer[1024];
    
    printf("Handling client: %s:%d\n", 
           inet_ntoa(client_addr->sin_addr), 
           ntohs(client_addr->sin_port));
    
    while (1) {
        ssize_t bytes_received = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
        
        if (bytes_received <= 0) {
            break;  // Client disconnected or error
        }
        
        buffer[bytes_received] = '\0';
        
        // Process request (echo in this example)
        if (send(client_fd, buffer, bytes_received, 0) < 0) {
            break;
        }
    }
    
    close_socket(client_fd);
    printf("Client disconnected\n");
}

void* worker_thread(void* arg) {
    thread_pool_t* pool = (thread_pool_t*)arg;
    work_item_t item;
    
    while (pool->running) {
        if (work_queue_pop(&pool->work_queue, &item) == 0) {
            handle_client_request(item.client_fd, &item.client_addr);
        }
    }
    
    return NULL;
}

thread_pool_t* create_thread_pool(int thread_count) {
    thread_pool_t* pool = malloc(sizeof(thread_pool_t));
    if (!pool) return NULL;
    
    pool->thread_count = (thread_count > MAX_THREADS) ? MAX_THREADS : thread_count;
    pool->running = 1;
    
    work_queue_init(&pool->work_queue);
    
    // Create worker threads
    for (int i = 0; i < pool->thread_count; i++) {
        if (pthread_create(&pool->threads[i], NULL, worker_thread, pool) != 0) {
            // Handle thread creation failure
            pool->thread_count = i;
            break;
        }
    }
    
    printf("Thread pool created with %d threads\n", pool->thread_count);
    return pool;
}

void destroy_thread_pool(thread_pool_t* pool) {
    if (!pool) return;
    
    // Signal shutdown
    pool->running = 0;
    
    pthread_mutex_lock(&pool->work_queue.mutex);
    pool->work_queue.shutdown = 1;
    pthread_cond_broadcast(&pool->work_queue.not_empty);
    pthread_mutex_unlock(&pool->work_queue.mutex);
    
    // Wait for all threads to finish
    for (int i = 0; i < pool->thread_count; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    
    work_queue_destroy(&pool->work_queue);
    free(pool);
}

void thread_pool_server(int port, int thread_count) {
    socket_t server_fd = create_tcp_socket();
    if (server_fd == INVALID_SOCKET_FD) return;
    
    set_socket_reuse_addr(server_fd);
    
    if (bind_socket(server_fd, NULL, port) < 0) {
        close_socket(server_fd);
        return;
    }
    
    if (listen(server_fd, 10) < 0) {
        close_socket(server_fd);
        return;
    }
    
    thread_pool_t* pool = create_thread_pool(thread_count);
    if (!pool) {
        close_socket(server_fd);
        return;
    }
    
    printf("Thread pool server listening on port %d\n", port);
    
    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        socket_t client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd == INVALID_SOCKET_FD) {
            continue;
        }
        
        work_item_t item = {client_fd, client_addr};
        if (work_queue_push(&pool->work_queue, item) < 0) {
            // Queue full or shutting down
            close_socket(client_fd);
        }
    }
    
    destroy_thread_pool(pool);
    close_socket(server_fd);
}
```

### 3. Proxy Patterns

#### HTTP Proxy Server
```c
#define MAX_HEADER_SIZE 8192
#define MAX_HOSTNAME_SIZE 256

typedef struct {
    char method[16];
    char url[1024];
    char hostname[MAX_HOSTNAME_SIZE];
    int port;
    char headers[MAX_HEADER_SIZE];
    int content_length;
} http_request_t;

int parse_http_request(const char* request, http_request_t* parsed) {
    // Simple HTTP request parser
    const char* line_end = strstr(request, "\r\n");
    if (!line_end) return -1;
    
    // Parse request line
    sscanf(request, "%15s %1023s", parsed->method, parsed->url);
    
    // Extract hostname and port from URL or Host header
    const char* host_header = strstr(request, "Host: ");
    if (host_header) {
        host_header += 6;  // Skip "Host: "
        const char* host_end = strstr(host_header, "\r\n");
        if (host_end) {
            int host_len = host_end - host_header;
            if (host_len < MAX_HOSTNAME_SIZE) {
                strncpy(parsed->hostname, host_header, host_len);
                parsed->hostname[host_len] = '\0';
                
                // Check for port in hostname
                char* colon = strchr(parsed->hostname, ':');
                if (colon) {
                    *colon = '\0';
                    parsed->port = atoi(colon + 1);
                } else {
                    parsed->port = 80;  // Default HTTP port
                }
            }
        }
    }
    
    // Find end of headers
    const char* headers_end = strstr(request, "\r\n\r\n");
    if (headers_end) {
        int headers_len = headers_end - request;
        if (headers_len < MAX_HEADER_SIZE) {
            strncpy(parsed->headers, request, headers_len);
            parsed->headers[headers_len] = '\0';
        }
    }
    
    // Parse content length
    const char* content_length_header = strstr(request, "Content-Length: ");
    if (content_length_header) {
        parsed->content_length = atoi(content_length_header + 16);
    } else {
        parsed->content_length = 0;
    }
    
    return 0;
}

void handle_proxy_client(socket_t client_fd) {
    char request_buffer[8192];
    ssize_t bytes_received = recv(client_fd, request_buffer, sizeof(request_buffer) - 1, 0);
    
    if (bytes_received <= 0) {
        close_socket(client_fd);
        return;
    }
    
    request_buffer[bytes_received] = '\0';
    
    http_request_t request;
    if (parse_http_request(request_buffer, &request) < 0) {
        const char* error_response = "HTTP/1.1 400 Bad Request\r\n\r\n";
        send(client_fd, error_response, strlen(error_response), 0);
        close_socket(client_fd);
        return;
    }
    
    printf("Proxying request to %s:%d\n", request.hostname, request.port);
    
    // Connect to target server
    socket_t server_fd = create_tcp_socket();
    if (server_fd == INVALID_SOCKET_FD) {
        const char* error_response = "HTTP/1.1 502 Bad Gateway\r\n\r\n";
        send(client_fd, error_response, strlen(error_response), 0);
        close_socket(client_fd);
        return;
    }
    
    if (connect_socket(server_fd, request.hostname, request.port) < 0) {
        const char* error_response = "HTTP/1.1 502 Bad Gateway\r\n\r\n";
        send(client_fd, error_response, strlen(error_response), 0);
        close_socket(server_fd);
        close_socket(client_fd);
        return;
    }
    
    // Forward request to server
    send(server_fd, request_buffer, bytes_received, 0);
    
    // Relay data between client and server
    fd_set read_fds;
    char relay_buffer[4096];
    
    while (1) {
        FD_ZERO(&read_fds);
        FD_SET(client_fd, &read_fds);
        FD_SET(server_fd, &read_fds);
        
        socket_t max_fd = (client_fd > server_fd) ? client_fd : server_fd;
        
        int activity = select(max_fd + 1, &read_fds, NULL, NULL, NULL);
        if (activity <= 0) break;
        
        if (FD_ISSET(client_fd, &read_fds)) {
            ssize_t bytes = recv(client_fd, relay_buffer, sizeof(relay_buffer), 0);
            if (bytes <= 0) break;
            if (send(server_fd, relay_buffer, bytes, 0) <= 0) break;
        }
        
        if (FD_ISSET(server_fd, &read_fds)) {
            ssize_t bytes = recv(server_fd, relay_buffer, sizeof(relay_buffer), 0);
            if (bytes <= 0) break;
            if (send(client_fd, relay_buffer, bytes, 0) <= 0) break;
        }
    }
    
    close_socket(server_fd);
    close_socket(client_fd);
}

void http_proxy_server(int port) {
    socket_t server_fd = create_tcp_socket();
    if (server_fd == INVALID_SOCKET_FD) return;
    
    set_socket_reuse_addr(server_fd);
    
    if (bind_socket(server_fd, NULL, port) < 0) {
        close_socket(server_fd);
        return;
    }
    
    if (listen(server_fd, 10) < 0) {
        close_socket(server_fd);
        return;
    }
    
    printf("HTTP proxy server listening on port %d\n", port);
    
    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        socket_t client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd == INVALID_SOCKET_FD) {
            continue;
        }
        
        // Handle client in separate thread (simplified)
        pid_t pid = fork();
        if (pid == 0) {
            close_socket(server_fd);
            handle_proxy_client(client_fd);
            exit(0);
        } else if (pid > 0) {
            close_socket(client_fd);
        }
    }
    
    close_socket(server_fd);
}
```

### 4. Reliability Patterns

#### Circuit Breaker Pattern
```c
typedef enum {
    CIRCUIT_CLOSED,
    CIRCUIT_OPEN,
    CIRCUIT_HALF_OPEN
} circuit_state_t;

typedef struct {
    circuit_state_t state;
    int failure_count;
    int failure_threshold;
    time_t last_failure_time;
    int timeout_seconds;
    int success_threshold;  // For half-open state
    int half_open_success_count;
} circuit_breaker_t;

circuit_breaker_t* create_circuit_breaker(int failure_threshold, int timeout_seconds) {
    circuit_breaker_t* cb = malloc(sizeof(circuit_breaker_t));
    if (!cb) return NULL;
    
    cb->state = CIRCUIT_CLOSED;
    cb->failure_count = 0;
    cb->failure_threshold = failure_threshold;
    cb->timeout_seconds = timeout_seconds;
    cb->success_threshold = 3;  // Require 3 successes to close circuit
    cb->half_open_success_count = 0;
    cb->last_failure_time = 0;
    
    return cb;
}

int circuit_breaker_call_allowed(circuit_breaker_t* cb) {
    time_t now = time(NULL);
    
    switch (cb->state) {
        case CIRCUIT_CLOSED:
            return 1;  // Allow call
            
        case CIRCUIT_OPEN:
            if ((now - cb->last_failure_time) >= cb->timeout_seconds) {
                cb->state = CIRCUIT_HALF_OPEN;
                cb->half_open_success_count = 0;
                return 1;  // Allow one call to test
            }
            return 0;  // Circuit is open, reject call
            
        case CIRCUIT_HALF_OPEN:
            return 1;  // Allow call to test service
    }
    
    return 0;
}

void circuit_breaker_on_success(circuit_breaker_t* cb) {
    switch (cb->state) {
        case CIRCUIT_CLOSED:
            cb->failure_count = 0;
            break;
            
        case CIRCUIT_HALF_OPEN:
            cb->half_open_success_count++;
            if (cb->half_open_success_count >= cb->success_threshold) {
                cb->state = CIRCUIT_CLOSED;
                cb->failure_count = 0;
                printf("Circuit breaker closed - service recovered\n");
            }
            break;
            
        case CIRCUIT_OPEN:
            // Should not happen
            break;
    }
}

void circuit_breaker_on_failure(circuit_breaker_t* cb) {
    cb->failure_count++;
    cb->last_failure_time = time(NULL);
    
    switch (cb->state) {
        case CIRCUIT_CLOSED:
            if (cb->failure_count >= cb->failure_threshold) {
                cb->state = CIRCUIT_OPEN;
                printf("Circuit breaker opened - service failing\n");
            }
            break;
            
        case CIRCUIT_HALF_OPEN:
            cb->state = CIRCUIT_OPEN;
            cb->half_open_success_count = 0;
            printf("Circuit breaker reopened - service still failing\n");
            break;
            
        case CIRCUIT_OPEN:
            // Already open
            break;
    }
}

// Example usage with resilient client
int resilient_send_with_circuit_breaker(resilient_client_t* client, 
                                       circuit_breaker_t* cb, 
                                       const void* data, size_t len) {
    if (!circuit_breaker_call_allowed(cb)) {
        printf("Circuit breaker is open - call rejected\n");
        return -1;
    }
    
    int result = resilient_send(client, data, len);
    
    if (result >= 0) {
        circuit_breaker_on_success(cb);
    } else {
        circuit_breaker_on_failure(cb);
    }
    
    return result;
}
```

## Pattern Selection Guidelines

### When to Use Each Pattern

| Pattern | Use Case | Advantages | Disadvantages |
|---------|----------|------------|---------------|
| Iterative Server | Simple, low-load applications | Simple to implement | Cannot handle concurrent clients |
| Thread-per-client | Moderate load, CPU-intensive tasks | Natural concurrency | High memory usage, scaling limits |
| Thread Pool | High load, controlled resources | Efficient resource usage | Queue management complexity |
| Event-driven | Very high concurrency | Excellent scalability | Complex programming model |
| Connection Pool | Frequent short connections | Reduced connection overhead | Memory usage, connection management |
| Circuit Breaker | Unreliable external services | Prevents cascade failures | May reject valid requests |

## Best Practices

### Client Patterns
1. **Implement retry logic** with exponential backoff
2. **Use connection pooling** for multiple concurrent requests
3. **Add heartbeat mechanisms** for long-lived connections
4. **Handle partial failures** gracefully

### Server Patterns
1. **Choose architecture** based on expected load
2. **Implement proper resource limits** (connections, threads, memory)
3. **Add monitoring and metrics** for performance tracking
4. **Handle graceful shutdown** properly

### Reliability Patterns
1. **Set appropriate timeouts** for all operations
2. **Implement circuit breakers** for external dependencies
3. **Add logging and monitoring** for failure detection
4. **Design for graceful degradation** when possible

## Assessment Checklist

- [ ] Can implement client reconnection strategies
- [ ] Understands different server architecture patterns
- [ ] Can build proxy and gateway applications
- [ ] Implements reliability patterns effectively
- [ ] Chooses appropriate patterns for different scenarios
- [ ] Handles failures and edge cases gracefully

## Next Steps

After mastering socket programming patterns:
- Explore microservices architecture patterns
- Study distributed systems patterns (consensus, replication)
- Learn about modern async/await patterns in languages like Rust, Go

## Resources

- "Pattern-Oriented Software Architecture" by Frank Buschmann
- "Enterprise Integration Patterns" by Gregor Hohpe
- "Building Microservices" by Sam Newman
- "Release It!" by Michael T. Nygard (for reliability patterns)

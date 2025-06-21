# High-Performance Server Project

*Last Updated: June 21, 2025*

## Project Overview

This project implements a high-performance, multi-threaded server capable of handling thousands of concurrent connections using various I/O models and optimization techniques. The server demonstrates advanced concepts like epoll, thread pools, connection pooling, and zero-copy techniques.

## Learning Objectives

- Understand different I/O models (blocking, non-blocking, async)
- Implement epoll-based event-driven server architecture
- Design and implement thread pool patterns
- Optimize memory usage and reduce system calls
- Handle high concurrency with efficient resource management
- Implement connection pooling and load balancing
- Practice performance tuning and profiling

## Architecture Overview

```
high_performance_server/
├── src/
│   ├── server.c                    # Main server implementation
│   ├── epoll_manager.c            # Epoll event management
│   ├── thread_pool.c              # Thread pool implementation
│   ├── connection_pool.c          # Connection pooling
│   ├── protocol_handler.c         # Protocol handling logic
│   ├── memory_pool.c              # Memory pool allocator
│   ├── stats_collector.c          # Performance statistics
│   ├── config.c                   # Configuration management
│   └── utils.c                    # Utility functions
├── include/
│   ├── server.h
│   ├── epoll_manager.h
│   ├── thread_pool.h
│   ├── connection_pool.h
│   ├── protocol_handler.h
│   ├── memory_pool.h
│   ├── stats_collector.h
│   ├── config.h
│   └── common.h
├── tests/
│   ├── load_test.c                # Load testing client
│   ├── benchmark.c                # Performance benchmarks
│   └── stress_test.sh             # Stress testing script
├── configs/
│   └── server.conf                # Server configuration
├── Makefile
└── README.md
```

## Core Headers

### Common Definitions (include/common.h)

```c
#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <sys/sendfile.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <pthread.h>
#include <atomic_ops.h>

// Server configuration limits
#define MAX_EVENTS 10000
#define MAX_CONNECTIONS 65536
#define MAX_THREADS 128
#define BUFFER_SIZE 8192
#define MAX_LISTEN_BACKLOG 1024
#define DEFAULT_PORT 8080

// Memory pool configuration
#define MEMORY_POOL_SIZE (1024 * 1024 * 16)  // 16MB
#define MEMORY_BLOCK_SIZE 4096

// Connection states
typedef enum {
    CONN_STATE_INVALID = 0,
    CONN_STATE_ACCEPTING,
    CONN_STATE_READING,
    CONN_STATE_PROCESSING,
    CONN_STATE_WRITING,
    CONN_STATE_KEEPALIVE,
    CONN_STATE_CLOSING
} connection_state_t;

// Protocol types
typedef enum {
    PROTOCOL_HTTP = 0,
    PROTOCOL_ECHO,
    PROTOCOL_CUSTOM
} protocol_type_t;

// Forward declarations
typedef struct connection connection_t;
typedef struct server_config server_config_t;
typedef struct server_stats server_stats_t;
typedef struct thread_pool thread_pool_t;
typedef struct memory_pool memory_pool_t;
typedef struct epoll_manager epoll_manager_t;

// Atomic operations wrappers
#define ATOMIC_INC(ptr) __sync_add_and_fetch(ptr, 1)
#define ATOMIC_DEC(ptr) __sync_sub_and_fetch(ptr, 1)
#define ATOMIC_ADD(ptr, val) __sync_add_and_fetch(ptr, val)
#define ATOMIC_CAS(ptr, old, new) __sync_bool_compare_and_swap(ptr, old, new)

#endif // COMMON_H
```

### Server Configuration (include/config.h)

```c
#ifndef CONFIG_H
#define CONFIG_H

#include "common.h"

typedef struct server_config {
    // Network settings
    int port;
    char bind_address[64];
    int listen_backlog;
    int tcp_nodelay;
    int tcp_keepalive;
    int reuse_addr;
    int reuse_port;
    
    // Threading settings
    int thread_pool_size;
    int max_connections;
    int connection_timeout;
    
    // Buffer settings
    int recv_buffer_size;
    int send_buffer_size;
    int max_request_size;
    
    // Protocol settings
    protocol_type_t protocol;
    int enable_pipelining;
    int max_keepalive_requests;
    int keepalive_timeout;
    
    // Performance settings
    int enable_sendfile;
    int enable_tcp_cork;
    int enable_memory_pool;
    int memory_pool_size;
    
    // Logging settings
    int log_level;
    char log_file[256];
    int enable_access_log;
    
    // Security settings
    int max_connections_per_ip;
    int enable_rate_limiting;
    int rate_limit_requests;
    int rate_limit_window;
} server_config_t;

// Function prototypes
server_config_t* load_config(const char* config_file);
void free_config(server_config_t* config);
void print_config(const server_config_t* config);
int validate_config(const server_config_t* config);

#endif // CONFIG_H
```

### Connection Management (include/connection_pool.h)

```c
#ifndef CONNECTION_POOL_H
#define CONNECTION_POOL_H

#include "common.h"

typedef struct connection {
    int fd;
    connection_state_t state;
    protocol_type_t protocol;
    
    // Network information
    struct sockaddr_in client_addr;
    char client_ip[INET_ADDRSTRLEN];
    int client_port;
    
    // Timing information
    time_t connect_time;
    time_t last_activity;
    time_t request_start_time;
    
    // Buffers
    char* read_buffer;
    char* write_buffer;
    size_t read_buffer_size;
    size_t write_buffer_size;
    size_t bytes_read;
    size_t bytes_written;
    size_t bytes_to_write;
    
    // Request/Response data
    char* request_data;
    size_t request_length;
    char* response_data;
    size_t response_length;
    
    // Keep-alive information
    int keepalive_requests;
    int keep_alive;
    
    // Processing state
    void* protocol_data;
    int (*process_callback)(struct connection*);
    
    // Statistics
    uint64_t total_bytes_read;
    uint64_t total_bytes_written;
    uint32_t total_requests;
    
    // Memory management
    struct connection* next_free;
    int ref_count;
} connection_t;

typedef struct connection_pool {
    connection_t* connections;
    connection_t* free_list;
    int max_connections;
    int active_connections;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} connection_pool_t;

// Function prototypes
connection_pool_t* create_connection_pool(int max_connections);
void destroy_connection_pool(connection_pool_t* pool);
connection_t* acquire_connection(connection_pool_t* pool);
void release_connection(connection_pool_t* pool, connection_t* conn);
void reset_connection(connection_t* conn);
void print_connection_stats(const connection_pool_t* pool);

#endif // CONNECTION_POOL_H
```

### Epoll Manager (include/epoll_manager.h)

```c
#ifndef EPOLL_MANAGER_H
#define EPOLL_MANAGER_H

#include "common.h"

typedef struct epoll_event_data {
    connection_t* conn;
    int fd;
    uint32_t events;
    void* user_data;
} epoll_event_data_t;

typedef struct epoll_manager {
    int epoll_fd;
    struct epoll_event* events;
    int max_events;
    int timeout;
    
    // Statistics
    uint64_t total_events;
    uint64_t read_events;
    uint64_t write_events;
    uint64_t error_events;
    uint64_t hangup_events;
    
    // Callbacks
    int (*on_read_ready)(connection_t* conn);
    int (*on_write_ready)(connection_t* conn);
    int (*on_error)(connection_t* conn);
    int (*on_hangup)(connection_t* conn);
} epoll_manager_t;

// Function prototypes
epoll_manager_t* create_epoll_manager(int max_events);
void destroy_epoll_manager(epoll_manager_t* mgr);
int epoll_add_connection(epoll_manager_t* mgr, connection_t* conn, uint32_t events);
int epoll_modify_connection(epoll_manager_t* mgr, connection_t* conn, uint32_t events);
int epoll_remove_connection(epoll_manager_t* mgr, connection_t* conn);
int epoll_wait_events(epoll_manager_t* mgr);
void set_epoll_callbacks(epoll_manager_t* mgr,
                        int (*on_read)(connection_t*),
                        int (*on_write)(connection_t*),
                        int (*on_error)(connection_t*),
                        int (*on_hangup)(connection_t*));

#endif // EPOLL_MANAGER_H
```

### Thread Pool (include/thread_pool.h)

```c
#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include "common.h"

typedef struct work_item {
    void (*function)(void* arg);
    void* argument;
    struct work_item* next;
} work_item_t;

typedef struct thread_pool {
    pthread_t* threads;
    int num_threads;
    
    // Work queue
    work_item_t* work_queue_head;
    work_item_t* work_queue_tail;
    int queue_size;
    int max_queue_size;
    
    // Synchronization
    pthread_mutex_t mutex;
    pthread_cond_t work_available;
    pthread_cond_t work_done;
    
    // State
    int shutdown;
    int active_threads;
    
    // Statistics
    uint64_t tasks_submitted;
    uint64_t tasks_completed;
    uint64_t tasks_rejected;
} thread_pool_t;

// Function prototypes
thread_pool_t* create_thread_pool(int num_threads, int max_queue_size);
void destroy_thread_pool(thread_pool_t* pool);
int thread_pool_submit(thread_pool_t* pool, void (*function)(void*), void* argument);
void thread_pool_wait_all(thread_pool_t* pool);
void print_thread_pool_stats(const thread_pool_t* pool);

#endif // THREAD_POOL_H
```

### Memory Pool (include/memory_pool.h)

```c
#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include "common.h"

typedef struct memory_block {
    size_t size;
    int in_use;
    struct memory_block* next;
    struct memory_block* prev;
} memory_block_t;

typedef struct memory_pool {
    void* pool_memory;
    size_t pool_size;
    memory_block_t* free_blocks;
    memory_block_t* used_blocks;
    
    // Statistics
    size_t total_allocated;
    size_t total_freed;
    size_t peak_usage;
    size_t current_usage;
    int allocation_count;
    int free_count;
    
    // Synchronization
    pthread_mutex_t mutex;
} memory_pool_t;

// Function prototypes
memory_pool_t* create_memory_pool(size_t pool_size);
void destroy_memory_pool(memory_pool_t* pool);
void* memory_pool_alloc(memory_pool_t* pool, size_t size);
void memory_pool_free(memory_pool_t* pool, void* ptr);
void print_memory_pool_stats(const memory_pool_t* pool);
void memory_pool_defragment(memory_pool_t* pool);

#endif // MEMORY_POOL_H
```

### Statistics Collector (include/stats_collector.h)

```c
#ifndef STATS_COLLECTOR_H
#define STATS_COLLECTOR_H

#include "common.h"

typedef struct server_stats {
    // Connection statistics
    uint64_t total_connections;
    uint64_t active_connections;
    uint64_t max_concurrent_connections;
    uint64_t refused_connections;
    uint64_t timeout_connections;
    
    // Request statistics
    uint64_t total_requests;
    uint64_t requests_per_second;
    uint64_t avg_request_time_ms;
    uint64_t max_request_time_ms;
    
    // Data transfer statistics
    uint64_t total_bytes_read;
    uint64_t total_bytes_written;
    uint64_t bytes_per_second_read;
    uint64_t bytes_per_second_written;
    
    // Error statistics
    uint64_t protocol_errors;
    uint64_t system_errors;
    uint64_t timeout_errors;
    
    // Performance statistics
    double cpu_usage;
    uint64_t memory_usage;
    uint64_t file_descriptors_used;
    
    // Timing
    time_t start_time;
    time_t last_update_time;
    
    // Thread synchronization
    pthread_mutex_t mutex;
} server_stats_t;

// Function prototypes
server_stats_t* create_stats_collector(void);
void destroy_stats_collector(server_stats_t* stats);
void update_connection_stats(server_stats_t* stats, int delta);
void update_request_stats(server_stats_t* stats, uint64_t request_time_ms);
void update_data_stats(server_stats_t* stats, uint64_t bytes_read, uint64_t bytes_written);
void update_error_stats(server_stats_t* stats, int error_type);
void print_server_stats(const server_stats_t* stats);
void reset_stats(server_stats_t* stats);

#endif // STATS_COLLECTOR_H
```

## Core Implementation

### Main Server (src/server.c)

```c
#include "server.h"
#include "config.h"
#include "epoll_manager.h"
#include "thread_pool.h"
#include "connection_pool.h"
#include "protocol_handler.h"
#include "memory_pool.h"
#include "stats_collector.h"

typedef struct server {
    server_config_t* config;
    int listen_fd;
    epoll_manager_t* epoll_mgr;
    thread_pool_t* thread_pool;
    connection_pool_t* conn_pool;
    memory_pool_t* mem_pool;
    server_stats_t* stats;
    
    volatile int running;
} server_t;

static server_t* g_server = NULL;

void signal_handler(int sig) {
    if (g_server) {
        printf("\nReceived signal %d, shutting down server...\n", sig);
        g_server->running = 0;
    }
}

void setup_signal_handlers(void) {
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);
}

int create_listen_socket(const server_config_t* config) {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd == -1) {
        perror("socket");
        return -1;
    }
    
    // Set socket options
    int opt = 1;
    if (config->reuse_addr) {
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    }
    
    if (config->reuse_port) {
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
    }
    
    if (config->tcp_nodelay) {
        setsockopt(listen_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    }
    
    // Set receive and send buffer sizes
    if (config->recv_buffer_size > 0) {
        setsockopt(listen_fd, SOL_SOCKET, SO_RCVBUF, 
                  &config->recv_buffer_size, sizeof(config->recv_buffer_size));
    }
    
    if (config->send_buffer_size > 0) {
        setsockopt(listen_fd, SOL_SOCKET, SO_SNDBUF, 
                  &config->send_buffer_size, sizeof(config->send_buffer_size));
    }
    
    // Bind socket
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(config->port);
    
    if (strlen(config->bind_address) > 0) {
        if (inet_pton(AF_INET, config->bind_address, &server_addr.sin_addr) <= 0) {
            fprintf(stderr, "Invalid bind address: %s\n", config->bind_address);
            close(listen_fd);
            return -1;
        }
    } else {
        server_addr.sin_addr.s_addr = INADDR_ANY;
    }
    
    if (bind(listen_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
        perror("bind");
        close(listen_fd);
        return -1;
    }
    
    // Start listening
    if (listen(listen_fd, config->listen_backlog) == -1) {
        perror("listen");
        close(listen_fd);
        return -1;
    }
    
    // Set non-blocking
    int flags = fcntl(listen_fd, F_GETFL, 0);
    if (fcntl(listen_fd, F_SETFL, flags | O_NONBLOCK) == -1) {
        perror("fcntl");
        close(listen_fd);
        return -1;
    }
    
    return listen_fd;
}

// Event handlers
int on_read_ready(connection_t* conn) {
    char buffer[BUFFER_SIZE];
    ssize_t bytes_read = recv(conn->fd, buffer, sizeof(buffer), 0);
    
    if (bytes_read > 0) {
        // Process received data
        update_data_stats(g_server->stats, bytes_read, 0);
        conn->last_activity = time(NULL);
        conn->total_bytes_read += bytes_read;
        
        // Append to connection buffer
        if (conn->bytes_read + bytes_read > conn->read_buffer_size) {
            // Reallocate buffer if needed
            size_t new_size = conn->read_buffer_size * 2;
            char* new_buffer = memory_pool_alloc(g_server->mem_pool, new_size);
            if (new_buffer) {
                memcpy(new_buffer, conn->read_buffer, conn->bytes_read);
                memory_pool_free(g_server->mem_pool, conn->read_buffer);
                conn->read_buffer = new_buffer;
                conn->read_buffer_size = new_size;
            } else {
                return -1; // Out of memory
            }
        }
        
        memcpy(conn->read_buffer + conn->bytes_read, buffer, bytes_read);
        conn->bytes_read += bytes_read;
        
        // Submit to thread pool for processing
        conn->state = CONN_STATE_PROCESSING;
        thread_pool_submit(g_server->thread_pool, 
                          (void(*)(void*))process_connection_data, conn);
        
        return 0;
    } else if (bytes_read == 0) {
        // Connection closed by client
        return -1;
    } else {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return 0; // No data available
        }
        return -1; // Error
    }
}

int on_write_ready(connection_t* conn) {
    if (conn->bytes_to_write > 0) {
        ssize_t bytes_written = send(conn->fd, 
                                   conn->write_buffer + conn->bytes_written,
                                   conn->bytes_to_write, 0);
        
        if (bytes_written > 0) {
            conn->bytes_written += bytes_written;
            conn->bytes_to_write -= bytes_written;
            conn->total_bytes_written += bytes_written;
            update_data_stats(g_server->stats, 0, bytes_written);
            
            if (conn->bytes_to_write == 0) {
                // All data sent
                if (conn->keep_alive) {
                    conn->state = CONN_STATE_KEEPALIVE;
                    // Modify epoll to only listen for read events
                    epoll_modify_connection(g_server->epoll_mgr, conn, EPOLLIN);
                } else {
                    conn->state = CONN_STATE_CLOSING;
                    return -1; // Close connection
                }
            }
            return 0;
        } else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                return 0; // Socket busy
            }
            return -1; // Error
        }
    }
    return 0;
}

int on_error(connection_t* conn) {
    update_error_stats(g_server->stats, 1);
    return -1;
}

int on_hangup(connection_t* conn) {
    return -1;
}

void accept_new_connections(server_t* server) {
    while (server->running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        int client_fd = accept(server->listen_fd, 
                              (struct sockaddr*)&client_addr, &client_len);
        
        if (client_fd == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break; // No more connections
            }
            perror("accept");
            continue;
        }
        
        // Set client socket non-blocking
        int flags = fcntl(client_fd, F_GETFL, 0);
        if (fcntl(client_fd, F_SETFL, flags | O_NONBLOCK) == -1) {
            perror("fcntl");
            close(client_fd);
            continue;
        }
        
        // Get connection from pool
        connection_t* conn = acquire_connection(server->conn_pool);
        if (!conn) {
            close(client_fd);
            update_error_stats(server->stats, 1);
            continue;
        }
        
        // Initialize connection
        conn->fd = client_fd;
        conn->state = CONN_STATE_READING;
        conn->protocol = server->config->protocol;
        conn->client_addr = client_addr;
        conn->connect_time = time(NULL);
        conn->last_activity = time(NULL);
        
        inet_ntop(AF_INET, &client_addr.sin_addr, conn->client_ip, INET_ADDRSTRLEN);
        conn->client_port = ntohs(client_addr.sin_port);
        
        // Add to epoll
        if (epoll_add_connection(server->epoll_mgr, conn, EPOLLIN | EPOLLET) == -1) {
            close(client_fd);
            release_connection(server->conn_pool, conn);
            continue;
        }
        
        update_connection_stats(server->stats, 1);
        
        printf("Accepted connection from %s:%d (fd=%d)\n", 
               conn->client_ip, conn->client_port, client_fd);
    }
}

void process_connection_data(connection_t* conn) {
    time_t start_time = time(NULL);
    
    // Process based on protocol
    int result = -1;
    switch (conn->protocol) {
        case PROTOCOL_HTTP:
            result = process_http_request(conn);
            break;
        case PROTOCOL_ECHO:
            result = process_echo_request(conn);
            break;
        case PROTOCOL_CUSTOM:
            result = process_custom_request(conn);
            break;
    }
    
    time_t end_time = time(NULL);
    uint64_t processing_time = (end_time - start_time) * 1000; // Convert to ms
    
    update_request_stats(g_server->stats, processing_time);
    
    if (result == 0) {
        // Data ready to send
        conn->state = CONN_STATE_WRITING;
        epoll_modify_connection(g_server->epoll_mgr, conn, EPOLLOUT | EPOLLET);
    } else {
        // Error or close connection
        conn->state = CONN_STATE_CLOSING;
        epoll_remove_connection(g_server->epoll_mgr, conn);
        close(conn->fd);
        release_connection(g_server->conn_pool, conn);
        update_connection_stats(g_server->stats, -1);
    }
}

void cleanup_timed_out_connections(server_t* server) {
    // This would be called periodically to clean up timed out connections
    // Implementation details omitted for brevity
}

void print_server_status(const server_t* server) {
    printf("\n=== Server Status ===\n");
    printf("Uptime: %ld seconds\n", time(NULL) - server->stats->start_time);
    print_server_stats(server->stats);
    print_connection_stats(server->conn_pool);
    print_thread_pool_stats(server->thread_pool);
    print_memory_pool_stats(server->mem_pool);
    printf("====================\n\n");
}

server_t* create_server(const char* config_file) {
    server_t* server = malloc(sizeof(server_t));
    if (!server) return NULL;
    
    memset(server, 0, sizeof(server_t));
    
    // Load configuration
    server->config = load_config(config_file);
    if (!server->config) {
        free(server);
        return NULL;
    }
    
    // Create listen socket
    server->listen_fd = create_listen_socket(server->config);
    if (server->listen_fd == -1) {
        free_config(server->config);
        free(server);
        return NULL;
    }
    
    // Create components
    server->epoll_mgr = create_epoll_manager(MAX_EVENTS);
    server->thread_pool = create_thread_pool(server->config->thread_pool_size, 1000);
    server->conn_pool = create_connection_pool(server->config->max_connections);
    server->mem_pool = create_memory_pool(server->config->memory_pool_size);
    server->stats = create_stats_collector();
    
    if (!server->epoll_mgr || !server->thread_pool || !server->conn_pool || 
        !server->mem_pool || !server->stats) {
        // Cleanup on failure
        if (server->epoll_mgr) destroy_epoll_manager(server->epoll_mgr);
        if (server->thread_pool) destroy_thread_pool(server->thread_pool);
        if (server->conn_pool) destroy_connection_pool(server->conn_pool);
        if (server->mem_pool) destroy_memory_pool(server->mem_pool);
        if (server->stats) destroy_stats_collector(server->stats);
        close(server->listen_fd);
        free_config(server->config);
        free(server);
        return NULL;
    }
    
    // Set epoll callbacks
    set_epoll_callbacks(server->epoll_mgr, on_read_ready, on_write_ready, 
                       on_error, on_hangup);
    
    // Add listen socket to epoll
    connection_t listen_conn = {0};
    listen_conn.fd = server->listen_fd;
    epoll_add_connection(server->epoll_mgr, &listen_conn, EPOLLIN);
    
    server->running = 1;
    
    return server;
}

void destroy_server(server_t* server) {
    if (!server) return;
    
    server->running = 0;
    
    if (server->listen_fd != -1) {
        close(server->listen_fd);
    }
    
    if (server->epoll_mgr) destroy_epoll_manager(server->epoll_mgr);
    if (server->thread_pool) destroy_thread_pool(server->thread_pool);
    if (server->conn_pool) destroy_connection_pool(server->conn_pool);
    if (server->mem_pool) destroy_memory_pool(server->mem_pool);
    if (server->stats) destroy_stats_collector(server->stats);
    if (server->config) free_config(server->config);
    
    free(server);
}

void run_server(server_t* server) {
    printf("High-Performance Server starting...\n");
    print_config(server->config);
    
    time_t last_status_print = time(NULL);
    
    while (server->running) {
        // Handle epoll events
        int num_events = epoll_wait_events(server->epoll_mgr);
        
        if (num_events > 0) {
            // Check if listen socket has events (new connections)
            accept_new_connections(server);
        }
        
        // Periodic maintenance
        time_t now = time(NULL);
        if (now - last_status_print >= 60) { // Print status every minute
            print_server_status(server);
            last_status_print = now;
        }
        
        // Clean up timed out connections
        cleanup_timed_out_connections(server);
    }
    
    printf("Server shutting down gracefully...\n");
    print_server_status(server);
}

int main(int argc, char* argv[]) {
    const char* config_file = (argc > 1) ? argv[1] : "configs/server.conf";
    
    setup_signal_handlers();
    
    server_t* server = create_server(config_file);
    if (!server) {
        fprintf(stderr, "Failed to create server\n");
        return 1;
    }
    
    g_server = server;
    run_server(server);
    
    destroy_server(server);
    return 0;
}
```

### Enhanced Makefile

```makefile
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -pthread -O3 -g -D_GNU_SOURCE
LDFLAGS = -pthread -latomic

SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin
TESTDIR = tests

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

# Test sources
TEST_SOURCES = $(wildcard $(TESTDIR)/*.c)
TEST_OBJECTS = $(TEST_SOURCES:$(TESTDIR)/%.c=$(OBJDIR)/%.o)

# Main target
TARGET = $(BINDIR)/high_performance_server
TEST_TARGETS = $(BINDIR)/load_test $(BINDIR)/benchmark

.PHONY: all clean test benchmark install

all: directories $(TARGET) $(TEST_TARGETS)

directories:
	@mkdir -p $(OBJDIR) $(BINDIR) configs

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

$(BINDIR)/load_test: $(OBJDIR)/load_test.o $(filter-out $(OBJDIR)/server.o, $(OBJECTS))
	$(CC) $^ -o $@ $(LDFLAGS)

$(BINDIR)/benchmark: $(OBJDIR)/benchmark.o $(filter-out $(OBJDIR)/server.o, $(OBJECTS))
	$(CC) $^ -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(OBJDIR)/%.o: $(TESTDIR)/%.c
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

test: $(BINDIR)/load_test
	./$(BINDIR)/load_test localhost 8080 100 1000

benchmark: $(BINDIR)/benchmark
	./$(BINDIR)/benchmark localhost 8080

install: $(TARGET)
	cp $(TARGET) /usr/local/bin/
	cp configs/server.conf /etc/

# Create default configuration
config:
	@echo "Creating default server configuration..."
	@mkdir -p configs
	@cat > configs/server.conf << 'EOF'
# High-Performance Server Configuration

# Network Settings
port = 8080
bind_address = 0.0.0.0
listen_backlog = 1024
tcp_nodelay = 1
tcp_keepalive = 1
reuse_addr = 1
reuse_port = 1

# Threading Settings
thread_pool_size = 16
max_connections = 10000
connection_timeout = 300

# Buffer Settings
recv_buffer_size = 65536
send_buffer_size = 65536
max_request_size = 1048576

# Protocol Settings
protocol = http
enable_pipelining = 1
max_keepalive_requests = 100
keepalive_timeout = 30

# Performance Settings
enable_sendfile = 1
enable_tcp_cork = 1
enable_memory_pool = 1
memory_pool_size = 16777216

# Logging Settings
log_level = 2
log_file = /var/log/server.log
enable_access_log = 1
EOF
```

## Performance Features

### Key Optimizations Implemented

1. **Zero-Copy I/O**: Uses `sendfile()` for static file serving
2. **Memory Pooling**: Custom memory allocator reduces malloc/free overhead
3. **Connection Pooling**: Reuses connection objects
4. **Thread Pool**: Avoids thread creation overhead
5. **Edge-Triggered Epoll**: Efficient event notification
6. **TCP Optimizations**: Nagle's algorithm control, TCP_CORK
7. **Buffer Management**: Optimized buffer sizing and reuse

### Usage Examples

```bash
# Build the server
make config  # Create default configuration
make         # Build server and tools

# Run the server
./bin/high_performance_server configs/server.conf

# Run load tests
./bin/load_test localhost 8080 1000 10000  # 1000 concurrent, 10000 requests

# Run benchmarks
./bin/benchmark localhost 8080
```

## Testing and Benchmarking

The project includes comprehensive testing tools:

- **Load Test Client**: Simulates high concurrent load
- **Benchmark Tool**: Measures performance metrics
- **Stress Test Scripts**: Automated stress testing

This high-performance server implementation demonstrates advanced socket programming concepts including event-driven architecture, efficient resource management, and performance optimization techniques essential for building scalable network applications.

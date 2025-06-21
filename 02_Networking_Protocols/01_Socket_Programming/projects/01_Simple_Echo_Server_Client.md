# Simple Echo Server/Client Project

*Last Updated: June 21, 2025*

## Project Overview

This project implements a TCP echo server and client that demonstrates fundamental socket programming concepts. The server accepts multiple client connections concurrently and echoes back any data received from clients.

## Learning Objectives

- Understand basic TCP socket operations
- Implement client-server communication
- Handle multiple concurrent clients
- Practice error handling and resource management
- Learn socket programming best practices

## Project Structure

```
echo_project/
├── src/
│   ├── echo_server.c
│   ├── echo_client.c
│   └── common.h
├── Makefile
└── README.md
```

## Implementation

### Common Header (common.h)

```c
#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>

#define DEFAULT_PORT 8080
#define BUFFER_SIZE 1024
#define MAX_CLIENTS 50
#define BACKLOG 10

// Function prototypes
void error_exit(const char* message);
void setup_signal_handlers(void);
void cleanup_resources(void);

// Global variables
extern volatile sig_atomic_t server_running;

#endif // COMMON_H
```

### Echo Server Implementation (echo_server.c)

```c
#include "common.h"

volatile sig_atomic_t server_running = 1;
int server_socket = -1;
pthread_mutex_t client_count_mutex = PTHREAD_MUTEX_INITIALIZER;
int active_clients = 0;

typedef struct {
    int client_socket;
    struct sockaddr_in client_addr;
    int client_id;
} client_info_t;

void error_exit(const char* message) {
    perror(message);
    cleanup_resources();
    exit(EXIT_FAILURE);
}

void signal_handler(int signal) {
    printf("\nReceived signal %d. Shutting down server gracefully...\n", signal);
    server_running = 0;
    
    if (server_socket != -1) {
        close(server_socket);
    }
}

void setup_signal_handlers(void) {
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    
    if (sigaction(SIGINT, &sa, NULL) == -1) {
        error_exit("sigaction(SIGINT)");
    }
    
    if (sigaction(SIGTERM, &sa, NULL) == -1) {
        error_exit("sigaction(SIGTERM)");
    }
    
    // Ignore SIGPIPE to handle broken pipe errors gracefully
    signal(SIGPIPE, SIG_IGN);
}

void cleanup_resources(void) {
    if (server_socket != -1) {
        close(server_socket);
        server_socket = -1;
    }
    
    pthread_mutex_destroy(&client_count_mutex);
    printf("Server resources cleaned up.\n");
}

void* handle_client(void* arg) {
    client_info_t* client = (client_info_t*)arg;
    char buffer[BUFFER_SIZE];
    ssize_t bytes_received, bytes_sent;
    
    printf("Client %d connected from %s:%d\n", 
           client->client_id,
           inet_ntoa(client->client_addr.sin_addr),
           ntohs(client->client_addr.sin_port));
    
    // Send welcome message
    const char* welcome = "Welcome to Echo Server! Type messages to echo them back.\n";
    send(client->client_socket, welcome, strlen(welcome), 0);
    
    while (server_running) {
        // Receive data from client
        bytes_received = recv(client->client_socket, buffer, BUFFER_SIZE - 1, 0);
        
        if (bytes_received <= 0) {
            if (bytes_received == 0) {
                printf("Client %d disconnected gracefully\n", client->client_id);
            } else {
                printf("Client %d disconnected with error: %s\n", 
                       client->client_id, strerror(errno));
            }
            break;
        }
        
        buffer[bytes_received] = '\0';
        printf("Client %d sent: %s", client->client_id, buffer);
        
        // Echo the message back to client
        bytes_sent = send(client->client_socket, buffer, bytes_received, 0);
        if (bytes_sent == -1) {
            printf("Failed to send echo to client %d: %s\n", 
                   client->client_id, strerror(errno));
            break;
        }
    }
    
    // Clean up client connection
    close(client->client_socket);
    
    pthread_mutex_lock(&client_count_mutex);
    active_clients--;
    printf("Client %d disconnected. Active clients: %d\n", 
           client->client_id, active_clients);
    pthread_mutex_unlock(&client_count_mutex);
    
    free(client);
    return NULL;
}

int create_server_socket(int port) {
    int sockfd;
    struct sockaddr_in server_addr;
    int opt = 1;
    
    // Create socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        error_exit("socket creation failed");
    }
    
    // Set socket options
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) == -1) {
        close(sockfd);
        error_exit("setsockopt failed");
    }
    
    // Configure server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);
    
    // Bind socket
    if (bind(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
        close(sockfd);
        error_exit("bind failed");
    }
    
    // Listen for connections
    if (listen(sockfd, BACKLOG) == -1) {
        close(sockfd);
        error_exit("listen failed");
    }
    
    printf("Echo server listening on port %d\n", port);
    printf("Maximum concurrent clients: %d\n", MAX_CLIENTS);
    printf("Press Ctrl+C to stop the server\n\n");
    
    return sockfd;
}

int main(int argc, char* argv[]) {
    int port = DEFAULT_PORT;
    int client_socket;
    struct sockaddr_in client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    pthread_t client_thread;
    int client_id = 1;
    
    // Parse command line arguments
    if (argc > 1) {
        port = atoi(argv[1]);
        if (port <= 0 || port > 65535) {
            fprintf(stderr, "Invalid port number: %s\n", argv[1]);
            exit(EXIT_FAILURE);
        }
    }
    
    // Setup signal handlers
    setup_signal_handlers();
    
    // Create and configure server socket
    server_socket = create_server_socket(port);
    
    // Main server loop
    while (server_running) {
        // Accept client connection
        client_socket = accept(server_socket, 
                              (struct sockaddr*)&client_addr, 
                              &client_addr_len);
        
        if (client_socket == -1) {
            if (errno == EINTR) {
                // Interrupted by signal, check if we should continue
                continue;
            }
            if (server_running) {
                perror("accept failed");
            }
            break;
        }
        
        // Check if we've reached the maximum number of clients
        pthread_mutex_lock(&client_count_mutex);
        if (active_clients >= MAX_CLIENTS) {
            pthread_mutex_unlock(&client_count_mutex);
            
            const char* reject_msg = "Server is full. Please try again later.\n";
            send(client_socket, reject_msg, strlen(reject_msg), 0);
            close(client_socket);
            
            printf("Rejected client connection - server full\n");
            continue;
        }
        active_clients++;
        pthread_mutex_unlock(&client_count_mutex);
        
        // Create client info structure
        client_info_t* client_info = malloc(sizeof(client_info_t));
        if (client_info == NULL) {
            perror("malloc failed");
            close(client_socket);
            
            pthread_mutex_lock(&client_count_mutex);
            active_clients--;
            pthread_mutex_unlock(&client_count_mutex);
            continue;
        }
        
        client_info->client_socket = client_socket;
        client_info->client_addr = client_addr;
        client_info->client_id = client_id++;
        
        // Create thread to handle client
        if (pthread_create(&client_thread, NULL, handle_client, client_info) != 0) {
            perror("pthread_create failed");
            close(client_socket);
            free(client_info);
            
            pthread_mutex_lock(&client_count_mutex);
            active_clients--;
            pthread_mutex_unlock(&client_count_mutex);
            continue;
        }
        
        // Detach thread so it cleans up automatically
        pthread_detach(client_thread);
    }
    
    printf("\nServer shutting down...\n");
    cleanup_resources();
    
    return 0;
}
```

### Echo Client Implementation (echo_client.c)

```c
#include "common.h"

volatile sig_atomic_t client_running = 1;
int client_socket = -1;

void signal_handler(int signal) {
    printf("\nReceived signal %d. Disconnecting from server...\n", signal);
    client_running = 0;
    
    if (client_socket != -1) {
        close(client_socket);
    }
}

void setup_signal_handlers(void) {
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    
    if (sigaction(SIGINT, &sa, NULL) == -1) {
        perror("sigaction(SIGINT)");
        exit(EXIT_FAILURE);
    }
    
    signal(SIGPIPE, SIG_IGN);
}

int connect_to_server(const char* server_ip, int port) {
    int sockfd;
    struct sockaddr_in server_addr;
    
    // Create socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        perror("socket creation failed");
        return -1;
    }
    
    // Configure server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    
    if (inet_pton(AF_INET, server_ip, &server_addr.sin_addr) <= 0) {
        fprintf(stderr, "Invalid server IP address: %s\n", server_ip);
        close(sockfd);
        return -1;
    }
    
    // Connect to server
    if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
        perror("connection failed");
        close(sockfd);
        return -1;
    }
    
    printf("Connected to server %s:%d\n", server_ip, port);
    return sockfd;
}

void* receive_messages(void* arg) {
    int sockfd = *(int*)arg;
    char buffer[BUFFER_SIZE];
    ssize_t bytes_received;
    
    while (client_running) {
        bytes_received = recv(sockfd, buffer, BUFFER_SIZE - 1, 0);
        
        if (bytes_received <= 0) {
            if (bytes_received == 0) {
                printf("\nServer closed the connection\n");
            } else if (errno != EINTR) {
                printf("\nConnection error: %s\n", strerror(errno));
            }
            client_running = 0;
            break;
        }
        
        buffer[bytes_received] = '\0';
        printf("Server: %s", buffer);
        fflush(stdout);
    }
    
    return NULL;
}

int main(int argc, char* argv[]) {
    char* server_ip = "127.0.0.1";
    int port = DEFAULT_PORT;
    char input_buffer[BUFFER_SIZE];
    pthread_t receive_thread;
    
    // Parse command line arguments
    if (argc > 1) {
        server_ip = argv[1];
    }
    if (argc > 2) {
        port = atoi(argv[2]);
        if (port <= 0 || port > 65535) {
            fprintf(stderr, "Invalid port number: %s\n", argv[2]);
            exit(EXIT_FAILURE);
        }
    }
    
    // Setup signal handlers
    setup_signal_handlers();
    
    // Connect to server
    client_socket = connect_to_server(server_ip, port);
    if (client_socket == -1) {
        exit(EXIT_FAILURE);
    }
    
    // Create thread to receive messages from server
    if (pthread_create(&receive_thread, NULL, receive_messages, &client_socket) != 0) {
        perror("pthread_create failed");
        close(client_socket);
        exit(EXIT_FAILURE);
    }
    
    printf("Type messages and press Enter to send them to the server.\n");
    printf("Type 'quit' to exit.\n\n");
    
    // Main input loop
    while (client_running) {
        printf("You: ");
        fflush(stdout);
        
        if (fgets(input_buffer, BUFFER_SIZE, stdin) == NULL) {
            if (feof(stdin)) {
                printf("\nEOF detected. Exiting...\n");
            }
            break;
        }
        
        // Check for quit command
        if (strncmp(input_buffer, "quit", 4) == 0) {
            printf("Disconnecting from server...\n");
            break;
        }
        
        // Send message to server
        ssize_t bytes_sent = send(client_socket, input_buffer, strlen(input_buffer), 0);
        if (bytes_sent == -1) {
            if (errno == EPIPE || errno == ECONNRESET) {
                printf("Connection lost\n");
            } else {
                perror("send failed");
            }
            break;
        }
    }
    
    client_running = 0;
    
    // Close socket to unblock receive thread
    if (client_socket != -1) {
        close(client_socket);
        client_socket = -1;
    }
    
    // Wait for receive thread to finish
    pthread_join(receive_thread, NULL);
    
    printf("Client terminated.\n");
    return 0;
}
```

### Makefile

```makefile
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -pthread -g
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Source files
SERVER_SRC = $(SRCDIR)/echo_server.c
CLIENT_SRC = $(SRCDIR)/echo_client.c

# Object files
SERVER_OBJ = $(OBJDIR)/echo_server.o
CLIENT_OBJ = $(OBJDIR)/echo_client.o

# Executables
SERVER_BIN = $(BINDIR)/echo_server
CLIENT_BIN = $(BINDIR)/echo_client

# Default target
all: directories $(SERVER_BIN) $(CLIENT_BIN)

# Create directories
directories:
	@mkdir -p $(OBJDIR) $(BINDIR)

# Build server
$(SERVER_BIN): $(SERVER_OBJ)
	$(CC) $(CFLAGS) -o $@ $<

# Build client
$(CLIENT_BIN): $(CLIENT_OBJ)
	$(CC) $(CFLAGS) -o $@ $<

# Compile server object
$(SERVER_OBJ): $(SERVER_SRC)
	$(CC) $(CFLAGS) -c -o $@ $<

# Compile client object
$(CLIENT_OBJ): $(CLIENT_SRC)
	$(CC) $(CFLAGS) -c -o $@ $<

# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Install (copy to system directories)
install: all
	sudo cp $(SERVER_BIN) /usr/local/bin/
	sudo cp $(CLIENT_BIN) /usr/local/bin/

# Uninstall
uninstall:
	sudo rm -f /usr/local/bin/echo_server
	sudo rm -f /usr/local/bin/echo_client

# Run server
run-server: $(SERVER_BIN)
	./$(SERVER_BIN)

# Run client
run-client: $(CLIENT_BIN)
	./$(CLIENT_BIN)

# Debug server
debug-server: $(SERVER_BIN)
	gdb ./$(SERVER_BIN)

# Debug client
debug-client: $(CLIENT_BIN)
	gdb ./$(CLIENT_BIN)

# Check for memory leaks
valgrind-server: $(SERVER_BIN)
	valgrind --leak-check=full --show-leak-kinds=all ./$(SERVER_BIN)

valgrind-client: $(CLIENT_BIN)
	valgrind --leak-check=full --show-leak-kinds=all ./$(CLIENT_BIN)

.PHONY: all directories clean install uninstall run-server run-client debug-server debug-client valgrind-server valgrind-client
```

## Usage Instructions

### Building the Project

```bash
# Build both server and client
make

# Build only server
make bin/echo_server

# Build only client
make bin/echo_client
```

### Running the Applications

**Terminal 1 (Server):**
```bash
# Run server on default port (8080)
./bin/echo_server

# Run server on custom port
./bin/echo_server 9000
```

**Terminal 2 (Client):**
```bash
# Connect to server on localhost:8080
./bin/echo_client

# Connect to specific server and port
./bin/echo_client 192.168.1.100 9000
```

### Testing Multiple Clients

Open multiple terminals and run the client in each:

```bash
# Terminal 2
./bin/echo_client

# Terminal 3
./bin/echo_client

# Terminal 4
./bin/echo_client
```

## Features Implemented

### Server Features
- ✅ Multi-threaded concurrent client handling
- ✅ Graceful shutdown with signal handling
- ✅ Client connection limits
- ✅ Resource cleanup and memory management
- ✅ Error handling and logging
- ✅ Welcome messages for new clients
- ✅ Client connection tracking

### Client Features
- ✅ Interactive command-line interface
- ✅ Threaded message receiving
- ✅ Graceful disconnection
- ✅ Signal handling for clean exit
- ✅ Connection error handling
- ✅ Real-time bidirectional communication

### Advanced Features
- ✅ Thread-safe client counting
- ✅ Proper socket option configuration
- ✅ Signal handling (SIGINT, SIGTERM, SIGPIPE)
- ✅ Memory leak prevention
- ✅ Cross-platform compatibility (Linux/Unix)
- ✅ Comprehensive error messages

## Testing Scenarios

### Basic Functionality Test
1. Start the server
2. Connect one client
3. Send messages and verify echo
4. Disconnect client gracefully

### Concurrent Clients Test
1. Start the server
2. Connect multiple clients (up to MAX_CLIENTS)
3. Send messages from different clients simultaneously
4. Verify each client receives only its own echoes

### Server Overload Test
1. Start the server
2. Try to connect more than MAX_CLIENTS
3. Verify excess clients are rejected with appropriate message

### Stress Test
1. Start the server
2. Connect maximum clients
3. Send high-frequency messages from all clients
4. Monitor server performance and memory usage

### Error Handling Test
1. Start server and client
2. Kill server process
3. Verify client detects disconnection
4. Test various network error scenarios

## Performance Considerations

- **Thread Pool**: Current implementation creates one thread per client. For higher loads, consider using a thread pool
- **Memory Usage**: Each client thread uses default stack size (~8MB on Linux)
- **File Descriptors**: Server limited by system's file descriptor limits
- **Buffer Sizes**: 1KB buffers suitable for text messages, increase for larger data

## Security Considerations

- **Input Validation**: Basic buffer overflow protection implemented
- **Resource Limits**: Client connection limits prevent resource exhaustion
- **Signal Handling**: Prevents zombie processes and ensures cleanup
- **Error Information**: Avoid revealing sensitive system information in error messages

## Possible Enhancements

1. **Configuration File**: Add server configuration via file
2. **Logging System**: Implement proper logging with levels
3. **SSL/TLS Support**: Add encryption for secure communication
4. **Authentication**: Add user authentication system
5. **Protocol Extensions**: Add custom protocol commands
6. **Performance Monitoring**: Add real-time performance metrics
7. **IPv6 Support**: Add IPv6 address support
8. **Cross-platform**: Add Windows support with Winsock

## Learning Outcomes

After completing this project, you should understand:

- Basic TCP socket programming concepts
- Client-server architecture design
- Multi-threaded programming with pthreads
- Signal handling in network applications
- Resource management and cleanup
- Error handling in network programming
- Build systems and project organization
- Testing and debugging network applications

This project provides a solid foundation for more advanced socket programming topics and serves as a reference implementation for basic TCP client-server communication.

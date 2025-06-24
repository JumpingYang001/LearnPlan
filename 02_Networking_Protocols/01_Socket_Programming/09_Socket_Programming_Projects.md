# Socket Programming Projects

*Last Updated: June 21, 2025*

## Overview

This module contains hands-on projects that combine the concepts learned throughout the Socket Programming track. These projects range from simple implementations to complex, real-world applications that demonstrate practical socket programming skills.

## Project Categories

### 1. Beginner Projects
- Simple Echo Server/Client
- Basic Chat Application
- File Transfer Utility

### 2. Intermediate Projects
- HTTP Server Implementation
- Multi-threaded Download Manager
- Network Monitoring Tool

### 3. Advanced Projects
- High-Performance Web Server
- Distributed Message Queue
- Network Protocol Analyzer

### 4. Specialized Projects
- IoT Device Simulator
- Game Server Architecture
- Real-time Data Streaming

## Detailed Project Specifications

### Project 1: Echo Server/Client (Beginner)

**Objective**: Build a robust TCP echo server and client with error handling.

#### What is an Echo Server?
An echo server is a simple network service that receives data from a client and sends the same data back. It's a classic beginner project for learning socket programming, covering the basics of network communication, concurrency, and error handling.

#### Key Concepts
- **Sockets**: Endpoints for communication between two machines.
- **TCP**: Reliable, connection-oriented protocol.
- **Concurrency**: Handling multiple clients at once (threads or processes).
- **Graceful Shutdown**: Properly closing sockets and cleaning up resources.

#### Requirements
- TCP server that echoes received messages back to clients
- Support multiple concurrent clients
- Proper error handling and logging
- Graceful shutdown mechanism
- Cross-platform compatibility

#### Implementation Steps
1. **Server Initialization and Binding**
2. **Accepting Client Connections**
3. **Message Echoing Logic**
4. **Concurrent Client Support (threading or forking)**
5. **Signal Handling for Graceful Shutdown**
6. **Comprehensive Error Reporting**

#### Example: Minimal TCP Echo Server (C, POSIX)
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>

void* handle_client(void* arg) {
    int client_sock = *(int*)arg;
    char buffer[1024];
    ssize_t bytes;
    while ((bytes = recv(client_sock, buffer, sizeof(buffer), 0)) > 0) {
        send(client_sock, buffer, bytes, 0); // Echo back
    }
    close(client_sock);
    return NULL;
}

int main() {
    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in server_addr = {0};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(12345);
    bind(server_sock, (struct sockaddr*)&server_addr, sizeof(server_addr));
    listen(server_sock, 5);
    printf("Echo server listening on port 12345...\n");
    while (1) {
        int client_sock = accept(server_sock, NULL, NULL);
        pthread_t tid;
        pthread_create(&tid, NULL, handle_client, &client_sock);
        pthread_detach(tid);
    }
    close(server_sock);
    return 0;
}
```

#### Example: Minimal TCP Echo Client (C, POSIX)
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

int main() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in server_addr = {0};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(12345);
    inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);
    connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr));
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), stdin)) {
        send(sock, buffer, strlen(buffer), 0);
        ssize_t bytes = recv(sock, buffer, sizeof(buffer)-1, 0);
        if (bytes <= 0) break;
        buffer[bytes] = '\0';
        printf("Echoed: %s", buffer);
    }
    close(sock);
    return 0;
}
```

#### Best Practices
- Always check return values of socket functions
- Use threads or processes for concurrency
- Handle signals for clean shutdown (e.g., SIGINT)
- Log errors and client connections

#### Debugging and Testing
- Use `telnet` or `nc` to test the server manually
- Add verbose logging for connections and errors
- Use `strace` or `lsof` to monitor socket usage

#### Extensions
- Add support for UDP echo
- Implement connection timeouts
- Add logging to a file
- Support IPv6

**Learning Outcomes**:
- Basic socket operations
- Server-client communication patterns
- Concurrent programming basics
- Error handling strategies

---

### Project 2: Multi-User Chat Application (Beginner-Intermediate)

**Objective**: Create a real-time chat system supporting multiple users.

#### What is a Multi-User Chat Application?
A chat application allows multiple users to communicate in real time via a central server. This project introduces concepts like message routing, user management, and protocol design.

#### Key Concepts
- **Centralized server**: All clients connect to a single server that routes messages.
- **Broadcast vs. private messaging**: Server distinguishes between messages for all and for specific users.
- **User management**: Track connected users, authentication, and presence.
- **Command parsing**: Support for commands like `/list`, `/msg`, `/quit`.

#### Requirements
- Central chat server
- Multiple chat clients
- User authentication (simple)
- Private messaging capability
- Broadcast messaging
- User list management
- Message history (optional)

#### Architecture Diagram
```
Client 1 ----\
              \
Client 2 ----> Chat Server ----> Message Broadcasting
              /                 /
Client N ----/                 /
                              /
                         Message History
                         User Management
```

#### Implementation Steps
1. **Design a simple text-based protocol** (e.g., `MSG`, `LOGIN`, `LIST`, `PRIVMSG`)
2. **Server accepts multiple clients** (use threads or select/poll)
3. **Maintain a user list** (username, socket, status)
4. **Broadcast and private message routing**
5. **Command parsing and response**
6. **Graceful client disconnect and cleanup**

#### Example: Chat Server Skeleton (C, POSIX, select-based)
```c
// Pseudocode for main server loop
fd_set master_set, read_fds;
int listener = socket(...);
bind(listener, ...);
listen(listener, ...);
FD_SET(listener, &master_set);
int fdmax = listener;
while (1) {
    read_fds = master_set;
    select(fdmax+1, &read_fds, NULL, NULL, NULL);
    for (int i = 0; i <= fdmax; i++) {
        if (FD_ISSET(i, &read_fds)) {
            if (i == listener) {
                // New client connection
                int newfd = accept(listener, ...);
                FD_SET(newfd, &master_set);
                if (newfd > fdmax) fdmax = newfd;
            } else {
                // Handle client message
                char buf[1024];
                int nbytes = recv(i, buf, sizeof(buf), 0);
                if (nbytes <= 0) {
                    // Client disconnected
                    close(i);
                    FD_CLR(i, &master_set);
                } else {
                    // Parse and route message
                }
            }
        }
    }
}
```

#### Example: Simple Protocol
```
LOGIN alice
MSG Hello everyone!
PRIVMSG bob Hi Bob!
LIST
QUIT
```

#### Best Practices
- Use non-blocking I/O or select/poll for scalability
- Validate all user input
- Use mutexes if using threads for shared data
- Log all connections and errors

#### Debugging and Testing
- Use multiple terminal windows to simulate clients
- Add verbose server logging
- Test edge cases (duplicate usernames, disconnects)

#### Extensions
- Add message history and offline messaging
- Implement a GUI client (e.g., with ncurses or Qt)
- Add encryption (TLS)

**Learning Outcomes**:
- Protocol design and parsing
- Multi-client server architecture
- User management and authentication
- Message routing and error handling

---

### Project 3: File Transfer Utility (Intermediate)


**Objective**: Build a reliable file transfer system over TCP/UDP.

#### What is a File Transfer Utility?
A file transfer utility enables sending and receiving files between computers over a network. This project covers protocol selection, chunked transfer, error recovery, and data integrity.

#### Key Concepts
- **TCP vs UDP**: TCP for reliability, UDP for speed (with custom reliability logic)
- **Chunking**: Split files into manageable pieces for transfer
- **Resumable Transfers**: Ability to continue after interruption
- **Checksums**: Ensure data integrity
- **Progress Reporting**: User feedback on transfer status

#### Requirements
- Support both TCP and UDP protocols
- Handle large files efficiently
- Resume interrupted transfers
- Progress reporting
- Integrity verification (checksums)
- Directory synchronization

#### Implementation Steps
1. **Protocol selection** (TCP/UDP)
2. **File chunking and reconstruction**
3. **Progress tracking and reporting**
4. **Checksum calculation and verification**
5. **Resume mechanism for interrupted transfers**
6. **Directory traversal and sync logic**

#### Example: TCP File Sender Skeleton (C)
```c
// Sender: send file in chunks
FILE* fp = fopen("file.txt", "rb");
char buffer[4096];
size_t n;
while ((n = fread(buffer, 1, sizeof(buffer), fp)) > 0) {
    send(sock, buffer, n, 0);
    // Optionally: send progress update
}
fclose(fp);
```

#### Example: TCP File Receiver Skeleton (C)
```c
// Receiver: write received chunks to file
FILE* fp = fopen("received.txt", "wb");
char buffer[4096];
ssize_t n;
while ((n = recv(sock, buffer, sizeof(buffer), 0)) > 0) {
    fwrite(buffer, 1, n, fp);
    // Optionally: update progress
}
fclose(fp);
```

#### Example: Simple Checksum (MD5, pseudo-code)
```c
#include <openssl/md5.h>
unsigned char md5[MD5_DIGEST_LENGTH];
MD5((unsigned char*)data, data_len, md5);
// Send/compare md5 for integrity
```

#### Best Practices
- Use non-blocking I/O for large files
- Always verify checksums after transfer
- Implement timeouts and retries for UDP
- Log transfer progress and errors

#### Debugging and Testing
- Use Wireshark to inspect file transfer packets
- Test with files of various sizes
- Simulate network interruptions to test resume

#### Extensions
- Add bandwidth throttling
- Support for secure transfer (TLS/DTLS)
- Implement directory sync (rsync-like)

**Learning Outcomes**:
- Reliable data transfer techniques
- Protocol selection and trade-offs
- Data integrity and error recovery
- Efficient file I/O and progress tracking

---

### Project 4: HTTP Server Implementation (Intermediate)

**Objective**: Build a functional HTTP/1.1 web server from scratch.

**Requirements**:
- HTTP/1.1 protocol compliance
- Static file serving
- Basic CGI support
- Virtual host support
- Access logging
- Configuration file support
- Security headers

**HTTP Features to Support**:
- GET, POST, HEAD methods
- Keep-alive connections
- Content-Type detection
- Range requests (partial content)
- Gzip compression
- Error pages (404, 500, etc.)

**Sample Configuration**:
```ini
# httpd.conf
port = 8080
document_root = /var/www/html
max_connections = 100
keep_alive_timeout = 30
enable_logging = true
log_file = /var/log/httpd.log
```

---

### Project 5: Network Monitoring Tool (Intermediate)

**Objective**: Create a network analysis and monitoring application.

**Requirements**:
- Packet capture using raw sockets
- Protocol analysis (TCP, UDP, ICMP)
- Network statistics collection
- Real-time traffic monitoring
- Alert system for anomalies
- Web-based dashboard (optional)

**Monitoring Capabilities**:
```c
// Key metrics to track:
1. Bandwidth utilization per interface
2. Connection counts and states
3. Packet loss and error rates
4. Top talkers (by traffic volume)
5. Protocol distribution
6. Security events (port scans, etc.)
```

---

### Project 6: High-Performance Web Server (Advanced)

**Objective**: Build a production-ready web server optimized for high concurrency.

**Requirements**:
- Event-driven architecture (epoll/kqueue/IOCP)
- HTTP/1.1 and HTTP/2 support
- SSL/TLS encryption
- Load balancing capabilities
- Caching layer
- Performance metrics and monitoring
- Configuration hot-reloading

**Performance Targets**:
- Handle 10,000+ concurrent connections (C10K problem)
- Sub-millisecond response times
- Memory usage under 1GB for 10K connections
- CPU utilization optimization

**Architecture Design**:
```
Load Balancer
     |
Event Loop (epoll/kqueue)
     |
+----+----+----+
|    |    |    |
Worker Threads
     |
Cache Layer
     |
Static Files / CGI / Reverse Proxy
```

---

### Project 7: Distributed Message Queue (Advanced)

**Objective**: Implement a distributed message queuing system.

**Requirements**:
- Multiple message brokers
- Topic-based messaging
- Persistence and durability
- Replication for high availability
- Consumer groups
- Message ordering guarantees
- Admin API and monitoring

**System Components**:
```
Producers --> Brokers (Cluster) --> Consumers
              |
              +-> Zookeeper/Raft (Coordination)
              |
              +-> Persistent Storage
```

**Message Features**:
- At-least-once delivery semantics
- Message partitioning for scalability
- Consumer offset management
- Dead letter queues
- Message TTL and expiration

---

### Project 8: Network Protocol Analyzer (Advanced)

**Objective**: Build a Wireshark-like network protocol analyzer.

**Requirements**:
- Packet capture from network interfaces
- Protocol dissection (Ethernet, IP, TCP, UDP, HTTP, etc.)
- Packet filtering and search
- Traffic analysis and statistics
- Export capabilities (PCAP format)
- Real-time and offline analysis

**Protocol Support**:
```c
// Protocol stack to implement:
Layer 2: Ethernet, WiFi frames
Layer 3: IPv4, IPv6, ICMP, ARP
Layer 4: TCP, UDP
Layer 7: HTTP, DNS, DHCP, SSH
```

**Analysis Features**:
- Connection tracking and state analysis
- Bandwidth analysis per conversation
- Protocol distribution charts
- Security analysis (suspicious patterns)
- Performance analysis (RTT, throughput)

---

### Project 9: IoT Device Simulator (Specialized)

**Objective**: Create a simulator for IoT devices and gateway.

**Requirements**:
- Simulate multiple IoT device types
- MQTT broker implementation
- CoAP protocol support
- Device management and provisioning
- Data aggregation and forwarding
- Security (TLS/DTLS)

**Device Types to Simulate**:
```c
// Different IoT device behaviors:
1. Temperature sensors (periodic data)
2. Motion detectors (event-driven)
3. Smart switches (command/response)
4. GPS trackers (location updates)
5. Industrial sensors (high-frequency data)
```

**Communication Patterns**:
- Publish/Subscribe (MQTT)
- Request/Response (CoAP)
- Device-to-device communication
- Cloud connectivity and synchronization

---

### Project 10: Game Server Architecture (Specialized)

**Objective**: Build a scalable multiplayer game server.

**Requirements**:
- Real-time message processing
- Player session management
- Game state synchronization
- Anti-cheat mechanisms
- Scalable architecture
- Matchmaking system
- Persistent player data

**Technical Challenges**:
```c
// Key areas to address:
1. Low-latency networking (UDP with reliability layer)
2. State synchronization across clients
3. Lag compensation and prediction
4. Scalability (horizontal partitioning)
5. Security (validation, encryption)
6. Monitoring and analytics
```

**Game Features**:
- Player authentication and profiles
- Room/lobby management
- Real-time gameplay communication
- Leaderboards and statistics
- Administrative tools

---

## Project Implementation Guidelines

### Development Phases

#### Phase 1: Core Implementation (40%)
- Basic functionality working
- Core protocols implemented
- Single-threaded/single-client support

#### Phase 2: Concurrency and Scaling (30%)
- Multi-client support
- Threading or event-driven architecture
- Basic error handling

#### Phase 3: Robustness and Features (20%)
- Comprehensive error handling
- Additional features
- Configuration and logging

#### Phase 4: Optimization and Polish (10%)
- Performance optimization
- Code cleanup and documentation
- Testing and validation

### Code Organization

```
project_name/
├── src/
│   ├── server/
│   ├── client/
│   ├── common/
│   └── protocols/
├── include/
├── tests/
├── docs/
├── configs/
├── scripts/
└── README.md
```

### Testing Strategy

1. **Unit Testing**
   - Test individual functions and modules
   - Mock network interfaces for testing
   - Validate protocol parsing and generation

2. **Integration Testing**
   - Test client-server communication
   - Validate end-to-end workflows
   - Test error conditions and recovery

3. **Performance Testing**
   - Load testing with multiple clients
   - Memory leak detection
   - Latency and throughput measurement

4. **Security Testing**
   - Input validation testing
   - Buffer overflow protection
   - Authentication and authorization

### Documentation Requirements

1. **README.md**
   - Project overview and objectives
   - Build and installation instructions
   - Usage examples and configuration
   - Known limitations and future work

2. **API Documentation**
   - Function and module documentation
   - Protocol specifications
   - Configuration options
   - Error codes and messages

3. **Architecture Documentation**
   - System design and components
   - Data flow diagrams
   - Threading and concurrency model
   - Performance characteristics

## Assessment Criteria

### Technical Implementation (60%)
- Code quality and organization
- Protocol correctness
- Error handling robustness
- Performance and scalability

### Features and Functionality (25%)
- Completeness of requirements
- Additional innovative features
- User experience quality
- Configuration flexibility

### Documentation and Testing (15%)
- Code documentation quality
- Test coverage and quality
- User documentation completeness
- Installation and setup ease

## Advanced Extensions

### Performance Optimization
- Implement zero-copy networking where possible
- Use memory pools for frequent allocations
- Optimize data structures for cache efficiency
- Profile and optimize hot code paths

### Security Enhancements
- Implement SSL/TLS encryption
- Add authentication and authorization
- Validate all input data
- Implement rate limiting and DDoS protection

### Monitoring and Observability
- Add comprehensive logging
- Implement metrics collection
- Create dashboards for monitoring
- Add distributed tracing support

### Cloud and Container Support
- Containerize applications (Docker)
- Add Kubernetes deployment configs
- Implement health checks and readiness probes
- Support cloud-native patterns

## Resources for Project Development

### Development Tools
- **Version Control**: Git with proper branching strategy
- **Build Systems**: CMake, Make, or platform-specific tools
- **Debugging**: GDB, Valgrind, AddressSanitizer
- **Profiling**: perf, gprof, Intel VTune

### Testing Tools
- **Load Testing**: Apache Bench (ab), wrk, JMeter
- **Network Simulation**: tc (traffic control), Mininet
- **Packet Analysis**: Wireshark, tcpdump
- **Memory Testing**: Valgrind, AddressSanitizer

### Documentation Tools
- **API Documentation**: Doxygen, Sphinx
- **Diagrams**: Graphviz, PlantUML, draw.io
- **README**: Markdown with proper formatting
- **Architecture**: C4 Model, UML diagrams

## Conclusion

These projects provide a comprehensive hands-on experience in socket programming, from basic concepts to advanced distributed systems. Each project builds upon previous knowledge while introducing new challenges and real-world scenarios.

The progression from simple echo servers to complex distributed systems mirrors the learning journey of a network programmer, providing practical experience with the concepts, patterns, and challenges encountered in professional software development.

Success in these projects demonstrates mastery of socket programming concepts and prepares you for advanced networking topics and professional development roles in system programming, network programming, and distributed systems.

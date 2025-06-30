# Transport Layer Protocols (TCP/UDP)

*Duration: 2 weeks*

## Overview

The Transport Layer (Layer 4 in the OSI model) is responsible for end-to-end communication between applications running on different hosts. It provides essential services like data delivery, error detection, flow control, and congestion control. The two primary protocols at this layer are TCP (Transmission Control Protocol) and UDP (User Datagram Protocol).

## Core Concepts

### Transport Layer Responsibilities

1. **End-to-end Communication**: Direct communication between applications
2. **Port-based Multiplexing**: Multiple applications can use the network simultaneously
3. **Error Detection and Recovery**: Ensuring data integrity
4. **Flow Control**: Managing data transmission rate
5. **Congestion Control**: Preventing network overload

### TCP vs UDP Comparison

| Feature | TCP | UDP |
|---------|-----|-----|
| **Connection** | Connection-oriented | Connectionless |
| **Reliability** | Reliable (guaranteed delivery) | Unreliable (best effort) |
| **Ordering** | Maintains packet order | No ordering guarantee |
| **Flow Control** | Yes | No |
| **Congestion Control** | Yes | No |
| **Header Size** | 20-60 bytes | 8 bytes |
| **Speed** | Slower (due to overhead) | Faster |
| **Use Cases** | Web browsing, email, file transfer | Gaming, streaming, DNS |

## TCP (Transmission Control Protocol)

### TCP Header Structure and Analysis

```c
#include <stdio.h>
#include <stdint.h>
#include <arpa/inet.h>

// Complete TCP header structure
struct tcp_header {
    uint16_t source;        // Source port (16 bits)
    uint16_t dest;          // Destination port (16 bits)
    uint32_t seq;           // Sequence number (32 bits)
    uint32_t ack_seq;       // Acknowledgment number (32 bits)
    uint16_t res1:4,        // Reserved (4 bits)
             doff:4,        // Data offset (4 bits) - header length
             fin:1,         // FIN flag
             syn:1,         // SYN flag
             rst:1,         // RST flag
             psh:1,         // PSH flag
             ack:1,         // ACK flag
             urg:1,         // URG flag
             ece:1,         // ECE flag
             cwr:1;         // CWR flag
    uint16_t window;        // Window size (16 bits)
    uint16_t check;         // Checksum (16 bits)
    uint16_t urg_ptr;       // Urgent pointer (16 bits)
    // Options can follow (0-40 bytes)
};

void print_tcp_header_info() {
    struct tcp_header tcp;
    
    printf("=== TCP Header Analysis ===\n");
    printf("Total header size: %zu bytes\n", sizeof(tcp));
    printf("Minimum header size: 20 bytes\n");
    printf("Maximum header size: 60 bytes (with options)\n\n");
    
    printf("Field breakdown:\n");
    printf("- Source port: %zu bytes\n", sizeof(tcp.source));
    printf("- Destination port: %zu bytes\n", sizeof(tcp.dest));
    printf("- Sequence number: %zu bytes\n", sizeof(tcp.seq));
    printf("- Acknowledgment: %zu bytes\n", sizeof(tcp.ack_seq));
    printf("- Flags: 2 bytes (includes data offset + flags)\n");
    printf("- Window size: %zu bytes\n", sizeof(tcp.window));
    printf("- Checksum: %zu bytes\n", sizeof(tcp.check));
    printf("- Urgent pointer: %zu bytes\n", sizeof(tcp.urg_ptr));
}

// Example: Parsing a TCP packet
void parse_tcp_packet(struct tcp_header* tcp_hdr) {
    printf("\n=== TCP Packet Analysis ===\n");
    printf("Source port: %u\n", ntohs(tcp_hdr->source));
    printf("Destination port: %u\n", ntohs(tcp_hdr->dest));
    printf("Sequence number: %u\n", ntohl(tcp_hdr->seq));
    printf("Acknowledgment: %u\n", ntohl(tcp_hdr->ack_seq));
    printf("Window size: %u bytes\n", ntohs(tcp_hdr->window));
    
    printf("Flags: ");
    if (tcp_hdr->syn) printf("SYN ");
    if (tcp_hdr->ack) printf("ACK ");
    if (tcp_hdr->fin) printf("FIN ");
    if (tcp_hdr->rst) printf("RST ");
    if (tcp_hdr->psh) printf("PSH ");
    if (tcp_hdr->urg) printf("URG ");
    printf("\n");
}

int main() {
    print_tcp_header_info();
    
    // Example TCP header (simulated)
    struct tcp_header example_tcp = {
        .source = htons(80),      // HTTP port
        .dest = htons(12345),     // Client port
        .seq = htonl(1000),
        .ack_seq = htonl(2000),
        .syn = 1,
        .ack = 1,
        .window = htons(8192)
    };
    
    parse_tcp_packet(&example_tcp);
    
    return 0;
}
```

### TCP Connection Management

#### Three-Way Handshake Process

The TCP three-way handshake establishes a reliable connection between client and server:

```
Client                          Server
  |                               |
  |  1. SYN (seq=x)              |
  |----------------------------->|
  |                               |
  |           2. SYN-ACK          |
  |     (seq=y, ack=x+1)         |
  |<-----------------------------|
  |                               |
  |  3. ACK (ack=y+1)            |
  |----------------------------->|
  |                               |
  |     Connection Established     |
```

**Implementation Example:**

```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>

// TCP Client implementing 3-way handshake
int tcp_client_connect(const char* server_ip, int port) {
    int sock_fd;
    struct sockaddr_in server_addr;
    
    // Step 1: Create socket
    sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        perror("Socket creation failed");
        return -1;
    }
    
    // Step 2: Configure server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    inet_pton(AF_INET, server_ip, &server_addr.sin_addr);
    
    // Step 3: Initiate connection (triggers 3-way handshake)
    printf("Initiating TCP connection to %s:%d\n", server_ip, port);
    if (connect(sock_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connection failed");
        close(sock_fd);
        return -1;
    }
    
    printf("TCP connection established successfully!\n");
    return sock_fd;
}

// TCP Server accepting connections
int tcp_server_listen(int port) {
    int server_fd, client_fd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);
    
    // Create socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("Socket creation failed");
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
    
    // Listen for connections
    if (listen(server_fd, 5) < 0) {
        perror("Listen failed");
        close(server_fd);
        return -1;
    }
    
    printf("TCP server listening on port %d\n", port);
    
    // Accept connection (completes 3-way handshake)
    client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
    if (client_fd < 0) {
        perror("Accept failed");
        close(server_fd);
        return -1;
    }
    
    printf("Client connected from %s\n", inet_ntoa(client_addr.sin_addr));
    
    close(server_fd); // Close listening socket
    return client_fd;
}
```

#### Connection Termination (Four-Way Handshake)

TCP connection termination requires a four-way handshake:

```
Client                          Server
  |                               |
  |  1. FIN (seq=x)              |
  |----------------------------->|
  |                               |
  |  2. ACK (ack=x+1)            |
  |<-----------------------------|
  |                               |
  |  3. FIN (seq=y)              |
  |<-----------------------------|
  |                               |
  |  4. ACK (ack=y+1)            |
  |----------------------------->|
  |                               |
  |     Connection Closed         |
```

### TCP Flow Control and Congestion Control

#### Window-based Flow Control

```c
#include <stdio.h>
#include <stdint.h>

// Simplified TCP window management
typedef struct {
    uint32_t send_base;      // Oldest unacknowledged sequence number
    uint32_t next_seq_num;   // Next sequence number to be sent
    uint16_t rwnd;           // Receiver window size
    uint16_t cwnd;           // Congestion window size
    uint16_t ssthresh;       // Slow start threshold
} tcp_window_t;

void tcp_window_update(tcp_window_t* window, uint32_t ack_num, uint16_t advertised_window) {
    // Update send base with acknowledged data
    if (ack_num > window->send_base) {
        window->send_base = ack_num;
        printf("ACK received: send_base updated to %u\n", window->send_base);
    }
    
    // Update receiver window
    window->rwnd = advertised_window;
    
    // Effective window = min(rwnd, cwnd)
    uint16_t effective_window = (window->rwnd < window->cwnd) ? window->rwnd : window->cwnd;
    
    printf("Effective window: %u bytes\n", effective_window);
    printf("Can send %u more bytes\n", 
           effective_window - (window->next_seq_num - window->send_base));
}

// Congestion control algorithms
void tcp_slow_start(tcp_window_t* window) {
    if (window->cwnd < window->ssthresh) {
        window->cwnd += 1;  // Exponential growth
        printf("Slow start: cwnd = %u\n", window->cwnd);
    }
}

void tcp_congestion_avoidance(tcp_window_t* window) {
    if (window->cwnd >= window->ssthresh) {
        window->cwnd += 1 / window->cwnd;  // Linear growth
        printf("Congestion avoidance: cwnd = %u\n", window->cwnd);
    }
}

void tcp_fast_recovery(tcp_window_t* window) {
    window->ssthresh = window->cwnd / 2;
    window->cwnd = window->ssthresh + 3;  // Fast recovery
    printf("Fast recovery: ssthresh = %u, cwnd = %u\n", 
           window->ssthresh, window->cwnd);
}
```

## UDP (User Datagram Protocol)

### UDP Header Structure and Analysis

```c
#include <stdio.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <string.h>

// UDP header structure (much simpler than TCP)
struct udp_header {
    uint16_t source;    // Source port (16 bits)
    uint16_t dest;      // Destination port (16 bits)
    uint16_t len;       // Length of UDP header + data (16 bits)
    uint16_t check;     // Checksum (16 bits)
};

void print_udp_header_info() {
    struct udp_header udp;
    
    printf("=== UDP Header Analysis ===\n");
    printf("Header size: %zu bytes (fixed)\n", sizeof(udp));
    printf("Minimum datagram size: 8 bytes (header only)\n");
    printf("Maximum datagram size: 65,535 bytes\n\n");
    
    printf("Field breakdown:\n");
    printf("- Source port: %zu bytes\n", sizeof(udp.source));
    printf("- Destination port: %zu bytes\n", sizeof(udp.dest));
    printf("- Length: %zu bytes\n", sizeof(udp.len));
    printf("- Checksum: %zu bytes\n", sizeof(udp.check));
}

// Calculate UDP checksum
uint16_t udp_checksum(struct udp_header* udp_hdr, char* data, int data_len) {
    // Simplified checksum calculation
    uint32_t sum = 0;
    uint16_t* ptr = (uint16_t*)udp_hdr;
    
    // Sum UDP header
    for (int i = 0; i < sizeof(struct udp_header) / 2; i++) {
        sum += ntohs(ptr[i]);
    }
    
    // Sum data (simplified - assuming even length)
    ptr = (uint16_t*)data;
    for (int i = 0; i < data_len / 2; i++) {
        sum += ntohs(ptr[i]);
    }
    
    // Fold 32-bit sum to 16 bits
    while (sum >> 16) {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    
    return ~sum;  // One's complement
}

// Create and send UDP datagram
void create_udp_datagram(const char* message) {
    struct udp_header udp;
    int message_len = strlen(message);
    
    // Fill UDP header
    udp.source = htons(12345);  // Source port
    udp.dest = htons(53);       // DNS port
    udp.len = htons(sizeof(udp) + message_len);
    udp.check = 0;  // Calculate later
    
    // Calculate checksum
    udp.check = udp_checksum(&udp, (char*)message, message_len);
    
    printf("\n=== UDP Datagram Created ===\n");
    printf("Source port: %u\n", ntohs(udp.source));
    printf("Destination port: %u\n", ntohs(udp.dest));
    printf("Total length: %u bytes\n", ntohs(udp.len));
    printf("Checksum: 0x%04x\n", ntohs(udp.check));
    printf("Payload: %s\n", message);
}

int main() {
    print_udp_header_info();
    create_udp_datagram("Hello, UDP!");
    return 0;
}
```

### UDP Socket Programming

#### UDP Client Example

```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>

int udp_client_example() {
    int sock_fd;
    struct sockaddr_in server_addr;
    char message[] = "Hello from UDP client!";
    char buffer[1024];
    socklen_t addr_len = sizeof(server_addr);
    
    // Create UDP socket
    sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_fd < 0) {
        perror("Socket creation failed");
        return -1;
    }
    
    // Configure server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(8080);
    inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);
    
    // Send data (no connection needed)
    printf("Sending UDP message: %s\n", message);
    ssize_t sent = sendto(sock_fd, message, strlen(message), 0,
                         (struct sockaddr*)&server_addr, sizeof(server_addr));
    
    if (sent < 0) {
        perror("Send failed");
        close(sock_fd);
        return -1;
    }
    
    // Receive response
    ssize_t received = recvfrom(sock_fd, buffer, sizeof(buffer) - 1, 0,
                               (struct sockaddr*)&server_addr, &addr_len);
    
    if (received > 0) {
        buffer[received] = '\0';
        printf("Received UDP response: %s\n", buffer);
    }
    
    close(sock_fd);
    return 0;
}
```

#### UDP Server Example

```c
int udp_server_example() {
    int sock_fd;
    struct sockaddr_in server_addr, client_addr;
    char buffer[1024];
    socklen_t client_len = sizeof(client_addr);
    
    // Create UDP socket
    sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_fd < 0) {
        perror("Socket creation failed");
        return -1;
    }
    
    // Configure server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(8080);
    
    // Bind socket
    if (bind(sock_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Bind failed");
        close(sock_fd);
        return -1;
    }
    
    printf("UDP server listening on port 8080\n");
    
    // Receive and echo messages
    while (1) {
        ssize_t received = recvfrom(sock_fd, buffer, sizeof(buffer) - 1, 0,
                                   (struct sockaddr*)&client_addr, &client_len);
        
        if (received > 0) {
            buffer[received] = '\0';
            printf("Received from %s: %s\n", 
                   inet_ntoa(client_addr.sin_addr), buffer);
            
            // Echo back to client
            sendto(sock_fd, buffer, received, 0,
                   (struct sockaddr*)&client_addr, client_len);
        }
    }
    
    close(sock_fd);
    return 0;
}
```

### UDP vs TCP Use Cases

#### When to Use UDP

**Real-time Applications:**
```c
// Gaming - position updates
struct player_position {
    uint32_t player_id;
    float x, y, z;
    uint32_t timestamp;
};

void send_position_update(int sock, struct sockaddr_in* server, 
                         struct player_position* pos) {
    // UDP is perfect here - if packet is lost, next update will come soon
    sendto(sock, pos, sizeof(*pos), 0, (struct sockaddr*)server, sizeof(*server));
}

// DNS queries - simple request/response
void dns_query_example() {
    // DNS uses UDP because:
    // 1. Simple request/response pattern
    // 2. Low latency required
    // 3. Can retry if needed
    // 4. Small packet size
}

// Media streaming
void stream_audio_packet(int sock, char* audio_data, int size) {
    // UDP streaming benefits:
    // - No head-of-line blocking
    // - Real-time delivery more important than reliability
    // - Can skip late packets
    sendto(sock, audio_data, size, 0, NULL, 0);
}
```

#### When to Use TCP

**Reliable Data Transfer:**
```c
// File transfer
int transfer_file_tcp(int sock, const char* filename) {
    FILE* file = fopen(filename, "rb");
    char buffer[1024];
    size_t bytes_read;
    
    while ((bytes_read = fread(buffer, 1, sizeof(buffer), file)) > 0) {
        // TCP ensures all data arrives in order
        send(sock, buffer, bytes_read, 0);
    }
    
    fclose(file);
    return 0;
}

// HTTP communication
void http_request_example(int sock) {
    char request[] = "GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n";
    
    // TCP is essential for HTTP because:
    // 1. Must receive complete response
    // 2. Order matters for HTML parsing
    // 3. No data loss acceptable
    send(sock, request, strlen(request), 0);
}

// Database connections
void database_transaction(int sock) {
    // TCP required for:
    // 1. ACID properties
    // 2. Reliable command delivery
    // 3. Consistent connection state
    char query[] = "SELECT * FROM users WHERE id = 1";
    send(sock, query, strlen(query), 0);
}
```

## Ports and Socket Programming

### Port Numbers and Well-Known Services

```c
#include <stdio.h>
#include <netdb.h>

// Common port assignments
typedef struct {
    uint16_t port;
    const char* service;
    const char* protocol;
    const char* description;
} port_info_t;

port_info_t well_known_ports[] = {
    {20, "FTP-DATA", "TCP", "File Transfer Protocol (Data)"},
    {21, "FTP", "TCP", "File Transfer Protocol (Control)"},
    {22, "SSH", "TCP", "Secure Shell"},
    {23, "TELNET", "TCP", "Telnet"},
    {25, "SMTP", "TCP", "Simple Mail Transfer Protocol"},
    {53, "DNS", "UDP/TCP", "Domain Name System"},
    {67, "DHCP", "UDP", "Dynamic Host Configuration Protocol (Server)"},
    {68, "DHCP", "UDP", "Dynamic Host Configuration Protocol (Client)"},
    {80, "HTTP", "TCP", "Hypertext Transfer Protocol"},
    {110, "POP3", "TCP", "Post Office Protocol v3"},
    {143, "IMAP", "TCP", "Internet Message Access Protocol"},
    {443, "HTTPS", "TCP", "HTTP Secure"},
    {993, "IMAPS", "TCP", "IMAP over SSL"},
    {995, "POP3S", "TCP", "POP3 over SSL"}
};

void print_port_info() {
    printf("=== Well-Known Ports (0-1023) ===\n");
    int num_ports = sizeof(well_known_ports) / sizeof(well_known_ports[0]);
    
    for (int i = 0; i < num_ports; i++) {
        printf("Port %d: %s (%s) - %s\n", 
               well_known_ports[i].port,
               well_known_ports[i].service,
               well_known_ports[i].protocol,
               well_known_ports[i].description);
    }
    
    printf("\nPort Ranges:\n");
    printf("- Well-Known Ports: 0-1023 (requires root privileges)\n");
    printf("- Registered Ports: 1024-49151 (assigned by IANA)\n");
    printf("- Dynamic/Private Ports: 49152-65535 (temporary use)\n");
}

// Get service information by port
void lookup_service_by_port(uint16_t port, const char* protocol) {
    struct servent* service = getservbyport(htons(port), protocol);
    
    if (service) {
        printf("Port %d/%s: %s\n", port, protocol, service->s_name);
        
        // Print aliases
        char** alias = service->s_aliases;
        if (*alias) {
            printf("Aliases: ");
            while (*alias) {
                printf("%s ", *alias);
                alias++;
            }
            printf("\n");
        }
    } else {
        printf("Port %d/%s: Unknown service\n", port, protocol);
    }
}

// Get port by service name
void lookup_port_by_service(const char* service, const char* protocol) {
    struct servent* service_info = getservbyname(service, protocol);
    
    if (service_info) {
        printf("Service %s/%s: Port %d\n", 
               service, protocol, ntohs(service_info->s_port));
    } else {
        printf("Service %s/%s: Not found\n", service, protocol);
    }
}
```

### Advanced Socket Programming

#### Socket Options and Configuration

```c
#include <sys/socket.h>
#include <netinet/tcp.h>

// Configure socket options for optimal performance
int configure_tcp_socket(int sock_fd) {
    int opt_val;
    socklen_t opt_len = sizeof(opt_val);
    
    // Enable address reuse (useful for servers)
    opt_val = 1;
    if (setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &opt_val, opt_len) < 0) {
        perror("SO_REUSEADDR failed");
        return -1;
    }
    
    // Set receive buffer size
    opt_val = 64 * 1024;  // 64KB
    if (setsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, &opt_val, opt_len) < 0) {
        perror("SO_RCVBUF failed");
        return -1;
    }
    
    // Set send buffer size
    opt_val = 64 * 1024;  // 64KB
    if (setsockopt(sock_fd, SOL_SOCKET, SO_SNDBUF, &opt_val, opt_len) < 0) {
        perror("SO_SNDBUF failed");
        return -1;
    }
    
    // Disable Nagle's algorithm for low latency
    opt_val = 1;
    if (setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, &opt_val, opt_len) < 0) {
        perror("TCP_NODELAY failed");
        return -1;
    }
    
    // Set keep-alive options
    opt_val = 1;
    if (setsockopt(sock_fd, SOL_SOCKET, SO_KEEPALIVE, &opt_val, opt_len) < 0) {
        perror("SO_KEEPALIVE failed");
        return -1;
    }
    
    // Keep-alive time (seconds before sending keep-alive probes)
    opt_val = 600;  // 10 minutes
    if (setsockopt(sock_fd, IPPROTO_TCP, TCP_KEEPIDLE, &opt_val, opt_len) < 0) {
        perror("TCP_KEEPIDLE failed");
        return -1;
    }
    
    // Keep-alive interval (seconds between keep-alive probes)
    opt_val = 60;   // 1 minute
    if (setsockopt(sock_fd, IPPROTO_TCP, TCP_KEEPINTVL, &opt_val, opt_len) < 0) {
        perror("TCP_KEEPINTVL failed");
        return -1;
    }
    
    // Keep-alive probe count
    opt_val = 3;
    if (setsockopt(sock_fd, IPPROTO_TCP, TCP_KEEPCNT, &opt_val, opt_len) < 0) {
        perror("TCP_KEEPCNT failed");
        return -1;
    }
    
    printf("Socket configured successfully\n");
    return 0;
}

// Get current socket options
void print_socket_info(int sock_fd) {
    int opt_val;
    socklen_t opt_len = sizeof(opt_val);
    
    printf("=== Socket Configuration ===\n");
    
    // Get socket type
    if (getsockopt(sock_fd, SOL_SOCKET, SO_TYPE, &opt_val, &opt_len) == 0) {
        switch (opt_val) {
            case SOCK_STREAM: printf("Socket type: TCP (SOCK_STREAM)\n"); break;
            case SOCK_DGRAM:  printf("Socket type: UDP (SOCK_DGRAM)\n"); break;
            default:          printf("Socket type: Other (%d)\n", opt_val); break;
        }
    }
    
    // Get buffer sizes
    if (getsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, &opt_val, &opt_len) == 0) {
        printf("Receive buffer size: %d bytes\n", opt_val);
    }
    
    if (getsockopt(sock_fd, SOL_SOCKET, SO_SNDBUF, &opt_val, &opt_len) == 0) {
        printf("Send buffer size: %d bytes\n", opt_val);
    }
    
    // Get TCP-specific options
    if (getsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, &opt_val, &opt_len) == 0) {
        printf("TCP_NODELAY: %s\n", opt_val ? "Enabled" : "Disabled");
    }
}
```

#### Error Handling and Timeout Management

```c
#include <errno.h>
#include <sys/select.h>

// Robust TCP send with timeout
ssize_t tcp_send_timeout(int sock_fd, const void* data, size_t len, int timeout_sec) {
    fd_set write_fds;
    struct timeval timeout;
    ssize_t total_sent = 0;
    ssize_t sent;
    
    while (total_sent < len) {
        // Set up file descriptor set
        FD_ZERO(&write_fds);
        FD_SET(sock_fd, &write_fds);
        
        // Set timeout
        timeout.tv_sec = timeout_sec;
        timeout.tv_usec = 0;
        
        // Wait for socket to be ready for writing
        int ready = select(sock_fd + 1, NULL, &write_fds, NULL, &timeout);
        
        if (ready < 0) {
            perror("select() failed");
            return -1;
        } else if (ready == 0) {
            printf("Send timeout after %d seconds\n", timeout_sec);
            errno = ETIMEDOUT;
            return -1;
        }
        
        // Send data
        sent = send(sock_fd, (char*)data + total_sent, len - total_sent, MSG_NOSIGNAL);
        
        if (sent < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;  // Try again
            } else {
                perror("send() failed");
                return -1;
            }
        } else if (sent == 0) {
            printf("Connection closed by peer\n");
            return total_sent;
        }
        
        total_sent += sent;
    }
    
    return total_sent;
}

// Robust TCP receive with timeout
ssize_t tcp_recv_timeout(int sock_fd, void* buffer, size_t len, int timeout_sec) {
    fd_set read_fds;
    struct timeval timeout;
    ssize_t received;
    
    // Set up file descriptor set
    FD_ZERO(&read_fds);
    FD_SET(sock_fd, &read_fds);
    
    // Set timeout
    timeout.tv_sec = timeout_sec;
    timeout.tv_usec = 0;
    
    // Wait for data to be available
    int ready = select(sock_fd + 1, &read_fds, NULL, NULL, &timeout);
    
    if (ready < 0) {
        perror("select() failed");
        return -1;
    } else if (ready == 0) {
        printf("Receive timeout after %d seconds\n", timeout_sec);
        errno = ETIMEDOUT;
        return -1;
    }
    
    // Receive data
    received = recv(sock_fd, buffer, len, 0);
    
    if (received < 0) {
        perror("recv() failed");
        return -1;
    } else if (received == 0) {
        printf("Connection closed by peer\n");
        return 0;
    }
    
    return received;
}

// Connection state monitoring
void monitor_connection_state(int sock_fd) {
    int error = 0;
    socklen_t len = sizeof(error);
    
    if (getsockopt(sock_fd, SOL_SOCKET, SO_ERROR, &error, &len) == 0) {
        if (error != 0) {
            printf("Socket error: %s\n", strerror(error));
        } else {
            printf("Socket is healthy\n");
        }
    } else {
        perror("getsockopt(SO_ERROR) failed");
    }
    
    // Check connection state (Linux-specific)
    #ifdef __linux__
    int state;
    len = sizeof(state);
    if (getsockopt(sock_fd, IPPROTO_TCP, TCP_INFO, &state, &len) == 0) {
        // TCP_INFO provides detailed connection statistics
        printf("TCP connection information available\n");
    }
    #endif
}
```

## Performance Optimization and Tuning

### TCP Performance Tuning

```c
// High-performance TCP server configuration
int create_high_performance_tcp_server(int port) {
    int server_fd;
    struct sockaddr_in server_addr;
    int opt = 1;
    
    // Create socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("Socket creation failed");
        return -1;
    }
    
    // Performance optimizations
    
    // 1. Enable address reuse
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    // 2. Enable port reuse (Linux)
    #ifdef SO_REUSEPORT
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
    #endif
    
    // 3. Increase socket buffers
    int buffer_size = 1024 * 1024;  // 1MB
    setsockopt(server_fd, SOL_SOCKET, SO_RCVBUF, &buffer_size, sizeof(buffer_size));
    setsockopt(server_fd, SOL_SOCKET, SO_SNDBUF, &buffer_size, sizeof(buffer_size));
    
    // 4. Disable Nagle's algorithm
    setsockopt(server_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    
    // 5. Enable TCP fast open (Linux)
    #ifdef TCP_FASTOPEN
    int queue_len = 100;
    setsockopt(server_fd, IPPROTO_TCP, TCP_FASTOPEN, &queue_len, sizeof(queue_len));
    #endif
    
    // 6. Set larger listen backlog
    int backlog = 1024;
    
    // Configure and bind
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);
    
    if (bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Bind failed");
        close(server_fd);
        return -1;
    }
    
    if (listen(server_fd, backlog) < 0) {
        perror("Listen failed");
        close(server_fd);
        return -1;
    }
    
    printf("High-performance TCP server listening on port %d\n", port);
    return server_fd;
}

// Measure network performance
typedef struct {
    double latency_ms;
    double throughput_mbps;
    int packets_sent;
    int packets_lost;
} network_stats_t;

network_stats_t measure_tcp_performance(int sock_fd, int test_duration_sec) {
    network_stats_t stats = {0};
    char buffer[1024];
    struct timeval start, end, current;
    double elapsed;
    
    gettimeofday(&start, NULL);
    
    while (1) {
        gettimeofday(&current, NULL);
        elapsed = (current.tv_sec - start.tv_sec) + 
                 (current.tv_usec - start.tv_usec) / 1000000.0;
        
        if (elapsed >= test_duration_sec) break;
        
        // Measure latency (echo test)
        struct timeval send_time, recv_time;
        gettimeofday(&send_time, NULL);
        
        if (send(sock_fd, &send_time, sizeof(send_time), 0) > 0) {
            if (recv(sock_fd, buffer, sizeof(buffer), 0) > 0) {
                gettimeofday(&recv_time, NULL);
                double latency = (recv_time.tv_sec - send_time.tv_sec) * 1000.0 +
                               (recv_time.tv_usec - send_time.tv_usec) / 1000.0;
                
                stats.latency_ms = (stats.latency_ms + latency) / 2;  // Running average
                stats.packets_sent++;
            } else {
                stats.packets_lost++;
            }
        }
        
        usleep(10000);  // 10ms between tests
    }
    
    // Calculate throughput (simplified)
    stats.throughput_mbps = (stats.packets_sent * sizeof(buffer) * 8) / 
                           (elapsed * 1000000);
    
    return stats;
}
```

## Learning Objectives

By the end of this section, you should be able to:

### Technical Understanding
- **Explain the differences** between TCP and UDP protocols with specific use cases
- **Analyze transport layer headers** and understand each field's purpose
- **Implement TCP connection management** including 3-way handshake and termination
- **Design UDP applications** for real-time communication scenarios
- **Configure socket options** for optimal performance
- **Handle network errors** and implement timeout mechanisms

### Practical Skills
- **Write TCP servers and clients** using socket programming
- **Implement UDP communication** for various application types
- **Debug network applications** using appropriate tools
- **Optimize network performance** through proper configuration
- **Design protocol selection strategies** based on application requirements

### Self-Assessment Checklist

Before proceeding to the next section, ensure you can:

□ Explain when to use TCP vs UDP for different applications  
□ Implement a complete TCP client-server application  
□ Create UDP-based real-time communication systems  
□ Configure socket options for performance optimization  
□ Handle connection failures and implement reconnection logic  
□ Analyze network traffic using packet capture tools  
□ Calculate TCP throughput and latency metrics  
□ Implement proper error handling for network operations  

## Study Materials and Resources

### Essential Reading
- **Primary:** "TCP/IP Illustrated, Volume 1" by W. Richard Stevens - Chapters 17-24
- **Alternative:** "Unix Network Programming" by W. Richard Stevens - Volumes 1 & 2
- **Online:** [RFC 793 (TCP)](https://tools.ietf.org/html/rfc793) and [RFC 768 (UDP)](https://tools.ietf.org/html/rfc768)
- **Reference:** Linux man pages: `socket(7)`, `tcp(7)`, `udp(7)`

### Practical Tools
```bash
# Network analysis tools
sudo apt-get install wireshark tcpdump netstat ss

# Performance testing
sudo apt-get install iperf3 netperf

# Development tools
sudo apt-get install build-essential gdb strace
```

### Hands-on Exercises

**Exercise 1: TCP Echo Server**
```c
// TODO: Implement a multi-threaded TCP echo server
// Requirements:
// - Handle multiple clients simultaneously
// - Implement proper error handling
// - Add connection logging
// - Support graceful shutdown
```

**Exercise 2: UDP Chat Application**
```c
// TODO: Create a UDP-based chat application
// Requirements:
// - Broadcast messages to multiple clients
// - Handle message ordering issues
// - Implement basic reliability mechanisms
// - Add timestamp to messages
```

**Exercise 3: Protocol Performance Comparison**
```c
// TODO: Compare TCP vs UDP performance
// Requirements:
// - Measure latency for both protocols
// - Test throughput under different conditions
// - Analyze packet loss scenarios
// - Generate performance report
```

### Debugging and Testing Commands

```bash
# Monitor network connections
netstat -tulpn                    # List listening ports
ss -tulpn                        # Modern alternative to netstat

# Capture network traffic
sudo tcpdump -i eth0 port 80     # Capture HTTP traffic
sudo wireshark                   # GUI packet analyzer

# Test connectivity
telnet hostname port             # Test TCP connection
nc -u hostname port              # Test UDP connection

# Performance testing
iperf3 -s                       # Start iperf3 server
iperf3 -c server_ip             # Run client test

# System limits
ulimit -n                       # Check file descriptor limit
cat /proc/sys/net/core/somaxconn # Check socket backlog limit
```

## Next Section
[Application Layer Protocols](05_Application_Layer.md)

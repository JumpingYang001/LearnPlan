# TCP/IP Implementation

*Duration: 2 weeks*

## Overview

TCP/IP implementation involves understanding how network protocols work at the system level, including socket programming, raw packet manipulation, protocol stack internals, and network buffer management. This section covers both high-level socket APIs and low-level packet crafting techniques.

## Learning Objectives

By the end of this section, you should be able to:
- **Implement socket-based network applications** using TCP and UDP
- **Create and manipulate raw packets** for custom protocol implementation
- **Understand TCP/IP stack internals** and how data flows through the network layers
- **Manage network buffers and timers** efficiently
- **Work with protocol control blocks** (PCBs) for connection management
- **Debug network applications** using appropriate tools
- **Implement custom protocols** on top of existing network infrastructure

## Topics Covered

### 1. Socket API Programming
### 2. Raw Socket Programming
### 3. TCP/IP Stack Internals
### 4. Buffer and Timer Management
### 5. Protocol Control Blocks
### 6. Network Performance Optimization

## 1. Socket API Programming

The Socket API provides a standardized interface for network communication. Understanding socket programming is fundamental to network application development.

### Socket Types and Protocols

#### TCP Sockets (SOCK_STREAM)
TCP provides reliable, connection-oriented communication with guaranteed delivery and ordering.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>

// TCP Server Implementation
int create_tcp_server(int port) {
    int server_fd, opt = 1;
    struct sockaddr_in address;
    
    // Create TCP socket
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        return -1;
    }
    
    // Set socket options to reuse address
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, 
                   &opt, sizeof(opt))) {
        perror("setsockopt");
        close(server_fd);
        return -1;
    }
    
    // Configure server address
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);
    
    // Bind socket to address
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        close(server_fd);
        return -1;
    }
    
    // Listen for connections (backlog = 10)
    if (listen(server_fd, 10) < 0) {
        perror("listen");
        close(server_fd);
        return -1;
    }
    
    printf("TCP Server listening on port %d\n", port);
    return server_fd;
}

// TCP Client Implementation
int create_tcp_client(const char* server_ip, int port) {
    int sock = 0;
    struct sockaddr_in serv_addr;
    
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("Socket creation error\n");
        return -1;
    }
    
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
    
    // Convert IPv4 address from text to binary
    if (inet_pton(AF_INET, server_ip, &serv_addr.sin_addr) <= 0) {
        printf("Invalid address/ Address not supported\n");
        close(sock);
        return -1;
    }
    
    // Connect to server
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("Connection Failed\n");
        close(sock);
        return -1;
    }
    
    printf("Connected to server %s:%d\n", server_ip, port);
    return sock;
}

// Complete TCP Echo Server Example
void tcp_echo_server_example() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    char buffer[1024] = {0};
    
    server_fd = create_tcp_server(8080);
    if (server_fd < 0) return;
    
    printf("Waiting for connections...\n");
    
    while (1) {
        // Accept incoming connection
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, 
                                (socklen_t*)&addrlen)) < 0) {
            perror("accept");
            continue;
        }
        
        printf("Connection accepted from %s:%d\n", 
               inet_ntoa(address.sin_addr), ntohs(address.sin_port));
        
        // Handle client communication
        while (1) {
            memset(buffer, 0, sizeof(buffer));
            ssize_t bytes_read = recv(new_socket, buffer, sizeof(buffer) - 1, 0);
            
            if (bytes_read <= 0) {
                if (bytes_read == 0) {
                    printf("Client disconnected\n");
                } else {
                    perror("recv");
                }
                break;
            }
            
            printf("Received: %s", buffer);
            
            // Echo back to client
            send(new_socket, buffer, bytes_read, 0);
        }
        
        close(new_socket);
    }
    
    close(server_fd);
}
```

#### UDP Sockets (SOCK_DGRAM)
UDP provides connectionless, unreliable communication with minimal overhead.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

// UDP Server Implementation
int create_udp_server(int port) {
    int sockfd;
    struct sockaddr_in servaddr;
    
    // Create UDP socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        return -1;
    }
    
    memset(&servaddr, 0, sizeof(servaddr));
    
    // Configure server address
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = INADDR_ANY;
    servaddr.sin_port = htons(port);
    
    // Bind socket to address
    if (bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
        perror("bind failed");
        close(sockfd);
        return -1;
    }
    
    printf("UDP Server listening on port %d\n", port);
    return sockfd;
}

// UDP Echo Server Example
void udp_echo_server_example() {
    int sockfd;
    char buffer[1024];
    struct sockaddr_in servaddr, cliaddr;
    socklen_t len = sizeof(cliaddr);
    
    sockfd = create_udp_server(8081);
    if (sockfd < 0) return;
    
    printf("UDP Server waiting for datagrams...\n");
    
    while (1) {
        memset(buffer, 0, sizeof(buffer));
        
        // Receive datagram from client
        ssize_t n = recvfrom(sockfd, buffer, sizeof(buffer) - 1, 0,
                            (struct sockaddr *)&cliaddr, &len);
        
        if (n < 0) {
            perror("recvfrom");
            continue;
        }
        
        printf("Received from %s:%d - %s", 
               inet_ntoa(cliaddr.sin_addr), ntohs(cliaddr.sin_port), buffer);
        
        // Echo back to client
        sendto(sockfd, buffer, n, 0, (const struct sockaddr *)&cliaddr, len);
    }
    
    close(sockfd);
}

// UDP Client Implementation
void udp_client_example(const char* server_ip, int port, const char* message) {
    int sockfd;
    char buffer[1024];
    struct sockaddr_in servaddr;
    
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        return;
    }
    
    memset(&servaddr, 0, sizeof(servaddr));
    
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(port);
    inet_pton(AF_INET, server_ip, &servaddr.sin_addr);
    
    // Send message to server
    sendto(sockfd, message, strlen(message), 0, 
           (const struct sockaddr *)&servaddr, sizeof(servaddr));
    
    // Receive echo from server
    socklen_t len = sizeof(servaddr);
    ssize_t n = recvfrom(sockfd, buffer, sizeof(buffer) - 1, 0,
                        (struct sockaddr *)&servaddr, &len);
    
    buffer[n] = '\0';
    printf("Server echo: %s\n", buffer);
    
    close(sockfd);
}
```

### Advanced Socket Options

```c
#include <sys/socket.h>
#include <netinet/tcp.h>

// Set various socket options for performance tuning
void configure_socket_options(int sockfd) {
    int opt = 1;
    
    // Enable address reuse
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    // Set keep-alive for TCP connections
    setsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE, &opt, sizeof(opt));
    
    // Disable Nagle's algorithm for low-latency applications
    setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    
    // Set socket buffer sizes
    int send_buffer = 65536;
    int recv_buffer = 65536;
    setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &send_buffer, sizeof(send_buffer));
    setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &recv_buffer, sizeof(recv_buffer));
    
    // Set timeout values
    struct timeval timeout;
    timeout.tv_sec = 5;
    timeout.tv_usec = 0;
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    
    printf("Socket options configured\n");
}
```

## 2. Raw Socket Programming

Raw sockets provide direct access to network protocols, allowing you to craft custom packets and implement custom protocols. This requires root privileges on most systems.

### Understanding Raw Sockets

Raw sockets bypass the transport layer and give you direct access to the IP layer, allowing you to:
- Create custom protocols
- Implement packet sniffing
- Perform network analysis
- Build security tools

### Raw Packet Creation and Injection

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <netinet/ip_icmp.h>
#include <arpa/inet.h>

// IP Header structure (if not available in headers)
struct ip_header {
    unsigned char  version_ihl;    // Version (4 bits) + IHL (4 bits)
    unsigned char  type_of_service; // Type of service
    unsigned short total_length;   // Total length
    unsigned short identification; // Identification
    unsigned short flags_fo;       // Flags (3 bits) + Fragment offset (13 bits)
    unsigned char  time_to_live;   // Time to live
    unsigned char  protocol;       // Protocol
    unsigned short header_checksum; // Header checksum
    unsigned int   source_address; // Source address
    unsigned int   dest_address;   // Destination address
};

// TCP Header structure
struct tcp_header {
    unsigned short source_port;     // Source port
    unsigned short dest_port;       // Destination port
    unsigned int   sequence;        // Sequence number
    unsigned int   acknowledge;     // Acknowledgement number
    unsigned char  ns : 1;          // NS flag
    unsigned char  reserved_part1 : 3; // Reserved
    unsigned char  data_offset : 4; // Data offset
    unsigned char  fin : 1;         // FIN flag
    unsigned char  syn : 1;         // SYN flag
    unsigned char  rst : 1;         // RST flag
    unsigned char  psh : 1;         // PSH flag
    unsigned char  ack : 1;         // ACK flag
    unsigned char  urg : 1;         // URG flag
    unsigned char  ecn : 1;         // ECN flag
    unsigned char  cwr : 1;         // CWR flag
    unsigned short window;          // Window
    unsigned short checksum;        // Checksum
    unsigned short urgent_pointer;  // Urgent pointer
};

// Calculate IP checksum
unsigned short calculate_checksum(unsigned short *ptr, int nbytes) {
    long sum;
    unsigned short oddbyte;
    short answer;

    sum = 0;
    while (nbytes > 1) {
        sum += *ptr++;
        nbytes -= 2;
    }
    
    if (nbytes == 1) {
        oddbyte = 0;
        *((u_char*)&oddbyte) = *(u_char*)ptr;
        sum += oddbyte;
    }

    sum = (sum >> 16) + (sum & 0xffff);
    sum = sum + (sum >> 16);
    answer = (short)~sum;
    
    return answer;
}

// Create and send a raw IP packet
void send_raw_ip_packet(const char* dest_ip, const char* source_ip, 
                       const char* data, int data_len) {
    int sock;
    struct sockaddr_in dest_addr;
    char packet[4096];
    struct ip_header *ip_hdr;
    int one = 1;
    const int *val = &one;
    
    // Create raw socket
    sock = socket(AF_INET, SOCK_RAW, IPPROTO_RAW);
    if (sock < 0) {
        perror("socket() error - need root privileges");
        return;
    }
    
    // Tell the kernel that headers are included in the packet
    if (setsockopt(sock, IPPROTO_IP, IP_HDRINCL, val, sizeof(one)) < 0) {
        perror("setsockopt() error");
        close(sock);
        return;
    }
    
    // Zero out the packet buffer
    memset(packet, 0, 4096);
    
    // IP header
    ip_hdr = (struct ip_header *)packet;
    ip_hdr->version_ihl = 0x45;        // Version 4, IHL 5 (20 bytes)
    ip_hdr->type_of_service = 0;       // TOS
    ip_hdr->total_length = htons(sizeof(struct ip_header) + data_len);
    ip_hdr->identification = htons(54321);
    ip_hdr->flags_fo = 0;              // No flags, no fragmentation
    ip_hdr->time_to_live = 255;        // TTL
    ip_hdr->protocol = IPPROTO_RAW;    // Protocol
    ip_hdr->header_checksum = 0;       // Will be calculated
    ip_hdr->source_address = inet_addr(source_ip);
    ip_hdr->dest_address = inet_addr(dest_ip);
    
    // Calculate IP header checksum
    ip_hdr->header_checksum = calculate_checksum((unsigned short *)packet, 
                                                sizeof(struct ip_header));
    
    // Copy data after IP header
    memcpy(packet + sizeof(struct ip_header), data, data_len);
    
    // Destination address
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = 0;
    dest_addr.sin_addr.s_addr = inet_addr(dest_ip);
    
    // Send the packet
    if (sendto(sock, packet, sizeof(struct ip_header) + data_len, 0,
               (struct sockaddr *)&dest_addr, sizeof(dest_addr)) < 0) {
        perror("sendto() error");
    } else {
        printf("Raw IP packet sent to %s\n", dest_ip);
    }
    
    close(sock);
}

// Create a TCP SYN packet for port scanning
void send_tcp_syn_packet(const char* dest_ip, int dest_port, const char* source_ip) {
    int sock;
    struct sockaddr_in dest_addr;
    char packet[4096];
    struct ip_header *ip_hdr;
    struct tcp_header *tcp_hdr;
    int one = 1;
    
    sock = socket(AF_INET, SOCK_RAW, IPPROTO_TCP);
    if (sock < 0) {
        perror("socket() error - need root privileges");
        return;
    }
    
    if (setsockopt(sock, IPPROTO_IP, IP_HDRINCL, &one, sizeof(one)) < 0) {
        perror("setsockopt() error");
        close(sock);
        return;
    }
    
    memset(packet, 0, 4096);
    
    // IP Header
    ip_hdr = (struct ip_header *)packet;
    ip_hdr->version_ihl = 0x45;
    ip_hdr->type_of_service = 0;
    ip_hdr->total_length = htons(sizeof(struct ip_header) + sizeof(struct tcp_header));
    ip_hdr->identification = htons(12345);
    ip_hdr->flags_fo = 0;
    ip_hdr->time_to_live = 64;
    ip_hdr->protocol = IPPROTO_TCP;
    ip_hdr->header_checksum = 0;
    ip_hdr->source_address = inet_addr(source_ip);
    ip_hdr->dest_address = inet_addr(dest_ip);
    
    // TCP Header
    tcp_hdr = (struct tcp_header *)(packet + sizeof(struct ip_header));
    tcp_hdr->source_port = htons(12345);
    tcp_hdr->dest_port = htons(dest_port);
    tcp_hdr->sequence = htonl(1000000);
    tcp_hdr->acknowledge = 0;
    tcp_hdr->data_offset = 5;  // TCP header size
    tcp_hdr->syn = 1;          // SYN flag
    tcp_hdr->window = htons(1024);
    tcp_hdr->checksum = 0;
    tcp_hdr->urgent_pointer = 0;
    
    // Calculate IP checksum
    ip_hdr->header_checksum = calculate_checksum((unsigned short *)packet, 
                                                sizeof(struct ip_header));
    
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = 0;
    dest_addr.sin_addr.s_addr = inet_addr(dest_ip);
    
    if (sendto(sock, packet, sizeof(struct ip_header) + sizeof(struct tcp_header), 0,
               (struct sockaddr *)&dest_addr, sizeof(dest_addr)) < 0) {
        perror("sendto() error");
    } else {
        printf("TCP SYN packet sent to %s:%d\n", dest_ip, dest_port);
    }
    
    close(sock);
}

// Packet sniffer using raw sockets
void packet_sniffer(const char* interface) {
    int sock;
    char buffer[65536];
    struct sockaddr saddr;
    int saddr_len = sizeof(saddr);
    
    // Create raw socket for sniffing
    sock = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (sock < 0) {
        perror("socket() error - need root privileges");
        return;
    }
    
    printf("Starting packet capture...\n");
    
    while (1) {
        // Receive packet
        ssize_t data_size = recvfrom(sock, buffer, 65536, 0, &saddr, 
                                    (socklen_t*)&saddr_len);
        
        if (data_size < 0) {
            perror("recvfrom() error");
            break;
        }
        
        // Process packet (simplified)
        struct ip_header *ip_hdr = (struct ip_header *)(buffer + 14); // Skip Ethernet header
        
        printf("Captured packet: %s -> %s, Protocol: %d, Size: %zd bytes\n",
               inet_ntoa(*(struct in_addr*)&ip_hdr->source_address),
               inet_ntoa(*(struct in_addr*)&ip_hdr->dest_address),
               ip_hdr->protocol, data_size);
    }
    
    close(sock);
}

// Example usage function
void raw_socket_examples() {
    printf("Raw Socket Programming Examples\n");
    printf("================================\n");
    
    // Example 1: Send raw IP packet
    printf("1. Sending raw IP packet...\n");
    send_raw_ip_packet("192.168.1.100", "192.168.1.1", "Hello Raw Socket!", 18);
    
    // Example 2: Send TCP SYN packet
    printf("2. Sending TCP SYN packet...\n");
    send_tcp_syn_packet("192.168.1.100", 80, "192.168.1.1");
    
    // Example 3: Start packet sniffer (uncomment to use)
    // printf("3. Starting packet sniffer...\n");
    // packet_sniffer("eth0");
}
```

### ICMP Implementation

```c
#include <netinet/ip_icmp.h>

// Send ICMP Echo Request (ping)
void send_ping(const char* dest_ip, const char* source_ip) {
    int sock;
    struct sockaddr_in dest_addr;
    char packet[4096];
    struct ip_header *ip_hdr;
    struct icmphdr *icmp_hdr;
    int one = 1;
    
    sock = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP);
    if (sock < 0) {
        perror("socket() error - need root privileges");
        return;
    }
    
    if (setsockopt(sock, IPPROTO_IP, IP_HDRINCL, &one, sizeof(one)) < 0) {
        perror("setsockopt() error");
        close(sock);
        return;
    }
    
    memset(packet, 0, 4096);
    
    // IP Header
    ip_hdr = (struct ip_header *)packet;
    ip_hdr->version_ihl = 0x45;
    ip_hdr->type_of_service = 0;
    ip_hdr->total_length = htons(sizeof(struct ip_header) + sizeof(struct icmphdr));
    ip_hdr->identification = htons(getpid());
    ip_hdr->flags_fo = 0;
    ip_hdr->time_to_live = 64;
    ip_hdr->protocol = IPPROTO_ICMP;
    ip_hdr->header_checksum = 0;
    ip_hdr->source_address = inet_addr(source_ip);
    ip_hdr->dest_address = inet_addr(dest_ip);
    
    // ICMP Header
    icmp_hdr = (struct icmphdr *)(packet + sizeof(struct ip_header));
    icmp_hdr->type = ICMP_ECHO;
    icmp_hdr->code = 0;
    icmp_hdr->checksum = 0;
    icmp_hdr->un.echo.id = getpid();
    icmp_hdr->un.echo.sequence = 1;
    
    // Calculate checksums
    icmp_hdr->checksum = calculate_checksum((unsigned short *)icmp_hdr, 
                                           sizeof(struct icmphdr));
    ip_hdr->header_checksum = calculate_checksum((unsigned short *)packet, 
                                                sizeof(struct ip_header));
    
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_addr.s_addr = inet_addr(dest_ip);
    
    if (sendto(sock, packet, sizeof(struct ip_header) + sizeof(struct icmphdr), 0,
               (struct sockaddr *)&dest_addr, sizeof(dest_addr)) < 0) {
        perror("sendto() error");
    } else {
        printf("ICMP Echo Request sent to %s\n", dest_ip);
    }
    
    close(sock);
}
```

## 3. TCP/IP Stack Internals

Understanding how the TCP/IP stack works internally helps you write more efficient network applications and debug network issues effectively.

### Network Stack Layers

```
┌─────────────────────────┐
│    Application Layer    │  ← Socket API (send/recv)
├─────────────────────────┤
│    Transport Layer      │  ← TCP/UDP (ports, reliability)
├─────────────────────────┤
│     Network Layer       │  ← IP (routing, addressing)
├─────────────────────────┤
│    Data Link Layer      │  ← Ethernet (MAC addresses)
├─────────────────────────┤
│    Physical Layer       │  ← Hardware (cables, radio)
└─────────────────────────┘
```

### TCP State Machine Implementation

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// TCP States
typedef enum {
    TCP_CLOSED,
    TCP_LISTEN,
    TCP_SYN_SENT,
    TCP_SYN_RCVD,
    TCP_ESTABLISHED,
    TCP_FIN_WAIT1,
    TCP_FIN_WAIT2,
    TCP_CLOSE_WAIT,
    TCP_CLOSING,
    TCP_LAST_ACK,
    TCP_TIME_WAIT
} tcp_state_t;

// TCP Connection structure
typedef struct {
    tcp_state_t state;
    uint32_t local_seq;
    uint32_t remote_seq;
    uint16_t local_port;
    uint16_t remote_port;
    uint32_t local_ip;
    uint32_t remote_ip;
    uint16_t window_size;
    uint8_t flags;
} tcp_connection_t;

// TCP Flags
#define TCP_FIN 0x01
#define TCP_SYN 0x02
#define TCP_RST 0x04
#define TCP_PSH 0x08
#define TCP_ACK 0x10
#define TCP_URG 0x20

const char* tcp_state_names[] = {
    "CLOSED", "LISTEN", "SYN_SENT", "SYN_RCVD", "ESTABLISHED",
    "FIN_WAIT1", "FIN_WAIT2", "CLOSE_WAIT", "CLOSING", "LAST_ACK", "TIME_WAIT"
};

// TCP State Machine
void tcp_state_machine(tcp_connection_t *conn, uint8_t flags, uint32_t seq, uint32_t ack) {
    printf("Current state: %s, Received flags: ", tcp_state_names[conn->state]);
    
    if (flags & TCP_SYN) printf("SYN ");
    if (flags & TCP_ACK) printf("ACK ");
    if (flags & TCP_FIN) printf("FIN ");
    if (flags & TCP_RST) printf("RST ");
    printf("\n");
    
    switch (conn->state) {
        case TCP_CLOSED:
            if (flags & TCP_SYN) {
                conn->state = TCP_SYN_RCVD;
                conn->remote_seq = seq;
                printf("State changed to: %s\n", tcp_state_names[conn->state]);
            }
            break;
            
        case TCP_LISTEN:
            if (flags & TCP_SYN) {
                conn->state = TCP_SYN_RCVD;
                conn->remote_seq = seq;
                printf("State changed to: %s\n", tcp_state_names[conn->state]);
            }
            break;
            
        case TCP_SYN_SENT:
            if ((flags & (TCP_SYN | TCP_ACK)) == (TCP_SYN | TCP_ACK)) {
                conn->state = TCP_ESTABLISHED;
                conn->remote_seq = seq;
                printf("State changed to: %s\n", tcp_state_names[conn->state]);
            } else if (flags & TCP_SYN) {
                conn->state = TCP_SYN_RCVD;
                conn->remote_seq = seq;
                printf("State changed to: %s\n", tcp_state_names[conn->state]);
            }
            break;
            
        case TCP_SYN_RCVD:
            if (flags & TCP_ACK) {
                conn->state = TCP_ESTABLISHED;
                printf("State changed to: %s\n", tcp_state_names[conn->state]);
            }
            break;
            
        case TCP_ESTABLISHED:
            if (flags & TCP_FIN) {
                conn->state = TCP_CLOSE_WAIT;
                printf("State changed to: %s\n", tcp_state_names[conn->state]);
            }
            break;
            
        case TCP_FIN_WAIT1:
            if (flags & TCP_ACK) {
                conn->state = TCP_FIN_WAIT2;
                printf("State changed to: %s\n", tcp_state_names[conn->state]);
            } else if (flags & TCP_FIN) {
                conn->state = TCP_CLOSING;
                printf("State changed to: %s\n", tcp_state_names[conn->state]);
            }
            break;
            
        case TCP_FIN_WAIT2:
            if (flags & TCP_FIN) {
                conn->state = TCP_TIME_WAIT;
                printf("State changed to: %s\n", tcp_state_names[conn->state]);
            }
            break;
            
        case TCP_CLOSE_WAIT:
            // Application should close, moving to LAST_ACK
            break;
            
        case TCP_CLOSING:
            if (flags & TCP_ACK) {
                conn->state = TCP_TIME_WAIT;
                printf("State changed to: %s\n", tcp_state_names[conn->state]);
            }
            break;
            
        case TCP_LAST_ACK:
            if (flags & TCP_ACK) {
                conn->state = TCP_CLOSED;
                printf("State changed to: %s\n", tcp_state_names[conn->state]);
            }
            break;
            
        case TCP_TIME_WAIT:
            // Wait for 2*MSL then go to CLOSED
            break;
    }
}

// Demonstrate TCP state transitions
void demonstrate_tcp_handshake() {
    tcp_connection_t client_conn = {TCP_CLOSED, 1000, 0, 12345, 80, 0, 0, 1024, 0};
    tcp_connection_t server_conn = {TCP_LISTEN, 2000, 0, 80, 12345, 0, 0, 1024, 0};
    
    printf("=== TCP Three-Way Handshake ===\n");
    
    // Step 1: Client sends SYN
    printf("1. Client -> Server: SYN\n");
    client_conn.state = TCP_SYN_SENT;
    tcp_state_machine(&server_conn, TCP_SYN, 1000, 0);
    
    // Step 2: Server sends SYN-ACK
    printf("2. Server -> Client: SYN-ACK\n");
    tcp_state_machine(&client_conn, TCP_SYN | TCP_ACK, 2000, 1001);
    
    // Step 3: Client sends ACK
    printf("3. Client -> Server: ACK\n");
    tcp_state_machine(&server_conn, TCP_ACK, 1001, 2001);
    
    printf("Connection established!\n\n");
}
```

### IP Routing Table Implementation

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>

// Routing table entry
typedef struct route_entry {
    uint32_t destination;    // Destination network
    uint32_t netmask;       // Network mask
    uint32_t gateway;       // Gateway IP
    char interface[16];     // Interface name
    int metric;             // Route metric
    struct route_entry *next;
} route_entry_t;

// Routing table
typedef struct {
    route_entry_t *entries;
    int count;
} routing_table_t;

// Initialize routing table
routing_table_t* init_routing_table() {
    routing_table_t *table = malloc(sizeof(routing_table_t));
    table->entries = NULL;
    table->count = 0;
    return table;
}

// Add route to table
void add_route(routing_table_t *table, const char *dest, const char *mask,
               const char *gateway, const char *interface, int metric) {
    route_entry_t *new_entry = malloc(sizeof(route_entry_t));
    
    new_entry->destination = inet_addr(dest);
    new_entry->netmask = inet_addr(mask);
    new_entry->gateway = inet_addr(gateway);
    strncpy(new_entry->interface, interface, sizeof(new_entry->interface) - 1);
    new_entry->metric = metric;
    new_entry->next = table->entries;
    
    table->entries = new_entry;
    table->count++;
    
    printf("Added route: %s/%s via %s dev %s metric %d\n",
           dest, mask, gateway, interface, metric);
}

// Find route for destination
route_entry_t* find_route(routing_table_t *table, uint32_t dest_ip) {
    route_entry_t *current = table->entries;
    route_entry_t *best_match = NULL;
    uint32_t longest_prefix = 0;
    
    while (current != NULL) {
        // Check if destination matches this route
        if ((dest_ip & current->netmask) == current->destination) {
            // Longest prefix match
            if (current->netmask > longest_prefix) {
                longest_prefix = current->netmask;
                best_match = current;
            }
        }
        current = current->next;
    }
    
    return best_match;
}

// Print routing table
void print_routing_table(routing_table_t *table) {
    printf("\nRouting Table:\n");
    printf("%-15s %-15s %-15s %-10s %-6s\n", 
           "Destination", "Netmask", "Gateway", "Interface", "Metric");
    printf("--------------------------------------------------------------------\n");
    
    route_entry_t *current = table->entries;
    while (current != NULL) {
        struct in_addr dest, mask, gw;
        dest.s_addr = current->destination;
        mask.s_addr = current->netmask;
        gw.s_addr = current->gateway;
        
        printf("%-15s %-15s %-15s %-10s %-6d\n",
               inet_ntoa(dest), inet_ntoa(mask), inet_ntoa(gw),
               current->interface, current->metric);
        
        current = current->next;
    }
}

// Example routing table usage
void routing_table_example() {
    routing_table_t *table = init_routing_table();
    
    // Add some routes
    add_route(table, "0.0.0.0", "0.0.0.0", "192.168.1.1", "eth0", 100);     // Default route
    add_route(table, "192.168.1.0", "255.255.255.0", "0.0.0.0", "eth0", 0); // Local network
    add_route(table, "10.0.0.0", "255.0.0.0", "192.168.1.254", "eth0", 50); // Private network
    
    print_routing_table(table);
    
    // Test route lookup
    uint32_t test_ip = inet_addr("192.168.1.100");
    route_entry_t *route = find_route(table, test_ip);
    
    if (route) {
        struct in_addr gw;
        gw.s_addr = route->gateway;
        printf("\nRoute for 192.168.1.100: via %s dev %s\n", 
               inet_ntoa(gw), route->interface);
    }
}
```

### Network Buffer Management

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Network buffer (sk_buff equivalent)
typedef struct net_buffer {
    unsigned char *data;        // Actual data
    unsigned char *head;        // Buffer start
    unsigned char *tail;        // Data end
    unsigned char *end;         // Buffer end
    int len;                    // Data length
    int truesize;              // True buffer size
    struct net_buffer *next;    // For buffer chains
} net_buffer_t;

// Buffer pool for efficient allocation
typedef struct {
    net_buffer_t **buffers;
    int size;
    int count;
    int free_count;
} buffer_pool_t;

// Create buffer pool
buffer_pool_t* create_buffer_pool(int size, int buffer_size) {
    buffer_pool_t *pool = malloc(sizeof(buffer_pool_t));
    pool->size = size;
    pool->count = 0;
    pool->free_count = size;
    
    pool->buffers = malloc(size * sizeof(net_buffer_t*));
    
    // Pre-allocate buffers
    for (int i = 0; i < size; i++) {
        net_buffer_t *buf = malloc(sizeof(net_buffer_t));
        buf->head = malloc(buffer_size);
        buf->data = buf->head;
        buf->tail = buf->head;
        buf->end = buf->head + buffer_size;
        buf->len = 0;
        buf->truesize = buffer_size;
        buf->next = NULL;
        
        pool->buffers[i] = buf;
    }
    
    printf("Created buffer pool with %d buffers of %d bytes each\n", size, buffer_size);
    return pool;
}

// Allocate buffer from pool
net_buffer_t* alloc_buffer(buffer_pool_t *pool) {
    if (pool->free_count == 0) {
        printf("Buffer pool exhausted!\n");
        return NULL;
    }
    
    net_buffer_t *buf = pool->buffers[--pool->free_count];
    buf->data = buf->head;
    buf->tail = buf->head;
    buf->len = 0;
    
    return buf;
}

// Free buffer back to pool
void free_buffer(buffer_pool_t *pool, net_buffer_t *buf) {
    if (pool->free_count < pool->size) {
        pool->buffers[pool->free_count++] = buf;
    }
}

// Add data to buffer
int buffer_put(net_buffer_t *buf, const void *data, int len) {
    if (buf->tail + len > buf->end) {
        printf("Buffer overflow!\n");
        return -1;
    }
    
    memcpy(buf->tail, data, len);
    buf->tail += len;
    buf->len += len;
    
    return len;
}

// Remove data from buffer
int buffer_pull(net_buffer_t *buf, void *data, int len) {
    if (buf->len < len) {
        len = buf->len;
    }
    
    if (data) {
        memcpy(data, buf->data, len);
    }
    
    buf->data += len;
    buf->len -= len;
    
    return len;
}

// Reserve space at buffer head
void buffer_reserve(net_buffer_t *buf, int len) {
    buf->data += len;
    buf->tail += len;
}

// Add protocol header
void* buffer_push(net_buffer_t *buf, int len) {
    buf->data -= len;
    buf->len += len;
    return buf->data;
}

// Example buffer usage
void buffer_management_example() {
    buffer_pool_t *pool = create_buffer_pool(100, 1500);
    
    // Allocate a buffer
    net_buffer_t *buf = alloc_buffer(pool);
    if (!buf) return;
    
    // Reserve space for headers
    buffer_reserve(buf, 64); // Reserve for Ethernet + IP + TCP headers
    
    // Add application data
    char app_data[] = "Hello, Network Buffer!";
    buffer_put(buf, app_data, strlen(app_data));
    
    // Add TCP header
    char *tcp_hdr = buffer_push(buf, 20);
    memset(tcp_hdr, 0, 20); // Simplified TCP header
    printf("Added TCP header at offset %ld\n", tcp_hdr - buf->head);
    
    // Add IP header
    char *ip_hdr = buffer_push(buf, 20);
    memset(ip_hdr, 0, 20); // Simplified IP header
    printf("Added IP header at offset %ld\n", ip_hdr - buf->head);
    
    // Add Ethernet header
    char *eth_hdr = buffer_push(buf, 14);
    memset(eth_hdr, 0, 14); // Simplified Ethernet header
    printf("Added Ethernet header at offset %ld\n", eth_hdr - buf->head);
    
    printf("Final packet size: %d bytes\n", buf->len);
    
    // Free buffer
    free_buffer(pool, buf);
}
```

## 4. Buffer and Timer Management

Efficient buffer and timer management is crucial for high-performance network applications. This section covers advanced techniques for managing network resources.

### Advanced Buffer Management

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <signal.h>
#include <unistd.h>

// Ring buffer for high-performance packet processing
typedef struct {
    unsigned char **buffers;    // Array of buffer pointers
    int *lengths;              // Length of each buffer
    int size;                  // Ring size (power of 2)
    int mask;                  // Size - 1 for fast modulo
    volatile int head;         // Producer index
    volatile int tail;         // Consumer index
} ring_buffer_t;

// Create ring buffer
ring_buffer_t* create_ring_buffer(int size, int buffer_size) {
    // Ensure size is power of 2
    int actual_size = 1;
    while (actual_size < size) actual_size <<= 1;
    
    ring_buffer_t *ring = malloc(sizeof(ring_buffer_t));
    ring->size = actual_size;
    ring->mask = actual_size - 1;
    ring->head = 0;
    ring->tail = 0;
    
    ring->buffers = malloc(actual_size * sizeof(unsigned char*));
    ring->lengths = malloc(actual_size * sizeof(int));
    
    // Pre-allocate buffers
    for (int i = 0; i < actual_size; i++) {
        ring->buffers[i] = malloc(buffer_size);
        ring->lengths[i] = 0;
    }
    
    printf("Created ring buffer: %d slots, %d bytes per buffer\n", 
           actual_size, buffer_size);
    return ring;
}

// Producer: Add data to ring buffer (lock-free)
int ring_buffer_put(ring_buffer_t *ring, const void *data, int len) {
    int current_head = ring->head;
    int next_head = (current_head + 1) & ring->mask;
    
    // Check if buffer is full
    if (next_head == ring->tail) {
        return -1; // Buffer full
    }
    
    // Copy data
    memcpy(ring->buffers[current_head], data, len);
    ring->lengths[current_head] = len;
    
    // Update head (atomic on x86)
    ring->head = next_head;
    
    return len;
}

// Consumer: Get data from ring buffer (lock-free)
int ring_buffer_get(ring_buffer_t *ring, void *data, int max_len) {
    int current_tail = ring->tail;
    
    // Check if buffer is empty
    if (current_tail == ring->head) {
        return 0; // Buffer empty
    }
    
    int len = ring->lengths[current_tail];
    if (len > max_len) len = max_len;
    
    // Copy data
    memcpy(data, ring->buffers[current_tail], len);
    
    // Update tail
    ring->tail = (current_tail + 1) & ring->mask;
    
    return len;
}

// Zero-copy buffer management
typedef struct zero_copy_buffer {
    unsigned char *data;
    int size;
    int ref_count;
    void (*destructor)(struct zero_copy_buffer*);
} zero_copy_buffer_t;

// Reference counting for zero-copy
zero_copy_buffer_t* buffer_get_ref(zero_copy_buffer_t *buf) {
    __sync_fetch_and_add(&buf->ref_count, 1);
    return buf;
}

void buffer_put_ref(zero_copy_buffer_t *buf) {
    if (__sync_sub_and_fetch(&buf->ref_count, 1) == 0) {
        if (buf->destructor) {
            buf->destructor(buf);
        }
        free(buf->data);
        free(buf);
    }
}

// Memory pool allocator
typedef struct memory_pool {
    void **free_list;
    int block_size;
    int total_blocks;
    int free_blocks;
    unsigned char *memory;
} memory_pool_t;

memory_pool_t* create_memory_pool(int block_size, int num_blocks) {
    memory_pool_t *pool = malloc(sizeof(memory_pool_t));
    
    pool->block_size = block_size;
    pool->total_blocks = num_blocks;
    pool->free_blocks = num_blocks;
    
    // Allocate contiguous memory
    pool->memory = malloc(block_size * num_blocks);
    pool->free_list = malloc(num_blocks * sizeof(void*));
    
    // Initialize free list
    for (int i = 0; i < num_blocks; i++) {
        pool->free_list[i] = pool->memory + (i * block_size);
    }
    
    printf("Created memory pool: %d blocks of %d bytes\n", num_blocks, block_size);
    return pool;
}

void* pool_alloc(memory_pool_t *pool) {
    if (pool->free_blocks == 0) {
        return NULL;
    }
    
    return pool->free_list[--pool->free_blocks];
}

void pool_free(memory_pool_t *pool, void *ptr) {
    if (pool->free_blocks < pool->total_blocks) {
        pool->free_list[pool->free_blocks++] = ptr;
    }
}
```

### Timer Management System

```c
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <signal.h>
#include <unistd.h>

// Timer callback function type
typedef void (*timer_callback_t)(void *arg);

// Timer structure
typedef struct timer {
    struct timeval expire_time;
    timer_callback_t callback;
    void *arg;
    int active;
    struct timer *next;
} timer_t;

// Timer wheel for efficient timer management
typedef struct {
    timer_t **slots;
    int size;
    int current_slot;
    struct timeval tick_interval;
} timer_wheel_t;

// Get current time
void get_current_time(struct timeval *tv) {
    gettimeofday(tv, NULL);
}

// Add milliseconds to timeval
void timeval_add_ms(struct timeval *tv, int ms) {
    tv->tv_usec += ms * 1000;
    if (tv->tv_usec >= 1000000) {
        tv->tv_sec += tv->tv_usec / 1000000;
        tv->tv_usec %= 1000000;
    }
}

// Compare timevals
int timeval_compare(const struct timeval *a, const struct timeval *b) {
    if (a->tv_sec < b->tv_sec) return -1;
    if (a->tv_sec > b->tv_sec) return 1;
    if (a->tv_usec < b->tv_usec) return -1;
    if (a->tv_usec > b->tv_usec) return 1;
    return 0;
}

// Create timer wheel
timer_wheel_t* create_timer_wheel(int size, int tick_ms) {
    timer_wheel_t *wheel = malloc(sizeof(timer_wheel_t));
    wheel->size = size;
    wheel->current_slot = 0;
    wheel->slots = calloc(size, sizeof(timer_t*));
    
    wheel->tick_interval.tv_sec = tick_ms / 1000;
    wheel->tick_interval.tv_usec = (tick_ms % 1000) * 1000;
    
    printf("Created timer wheel: %d slots, %d ms per tick\n", size, tick_ms);
    return wheel;
}

// Add timer to wheel
timer_t* add_timer(timer_wheel_t *wheel, int timeout_ms, 
                   timer_callback_t callback, void *arg) {
    timer_t *timer = malloc(sizeof(timer_t));
    
    get_current_time(&timer->expire_time);
    timeval_add_ms(&timer->expire_time, timeout_ms);
    
    timer->callback = callback;
    timer->arg = arg;
    timer->active = 1;
    
    // Calculate slot (simplified)
    int slot = (wheel->current_slot + (timeout_ms / 10)) % wheel->size;
    timer->next = wheel->slots[slot];
    wheel->slots[slot] = timer;
    
    return timer;
}

// Process expired timers
void process_timers(timer_wheel_t *wheel) {
    struct timeval now;
    get_current_time(&now);
    
    timer_t *current = wheel->slots[wheel->current_slot];
    timer_t *prev = NULL;
    
    while (current) {
        if (current->active && timeval_compare(&now, &current->expire_time) >= 0) {
            // Timer expired
            current->callback(current->arg);
            current->active = 0;
            
            // Remove from list
            if (prev) {
                prev->next = current->next;
            } else {
                wheel->slots[wheel->current_slot] = current->next;
            }
            
            timer_t *expired = current;
            current = current->next;
            free(expired);
        } else {
            prev = current;
            current = current->next;
        }
    }
    
    wheel->current_slot = (wheel->current_slot + 1) % wheel->size;
}

// Timer callback examples
void timeout_callback(void *arg) {
    int *id = (int*)arg;
    printf("Timer %d expired!\n", *id);
}

void retransmit_callback(void *arg) {
    char *packet = (char*)arg;
    printf("Retransmitting packet: %s\n", packet);
}

// Hierarchical timing wheels for different time scales
typedef struct {
    timer_wheel_t *ms_wheel;    // Millisecond precision
    timer_wheel_t *sec_wheel;   // Second precision
    timer_wheel_t *min_wheel;   // Minute precision
} hierarchical_timer_t;

hierarchical_timer_t* create_hierarchical_timer() {
    hierarchical_timer_t *htimer = malloc(sizeof(hierarchical_timer_t));
    
    htimer->ms_wheel = create_timer_wheel(1000, 1);    // 1ms ticks
    htimer->sec_wheel = create_timer_wheel(60, 1000);  // 1s ticks
    htimer->min_wheel = create_timer_wheel(60, 60000); // 1min ticks
    
    return htimer;
}

// TCP timer management example
typedef struct tcp_timers {
    timer_t *retransmit_timer;
    timer_t *keepalive_timer;
    timer_t *time_wait_timer;
    timer_wheel_t *wheel;
} tcp_timers_t;

void tcp_retransmit_timeout(void *arg) {
    printf("TCP retransmit timeout - resending packet\n");
    // Implement retransmission logic
}

void tcp_keepalive_timeout(void *arg) {
    printf("TCP keepalive timeout - sending probe\n");
    // Send keepalive probe
}

void tcp_time_wait_timeout(void *arg) {
    printf("TCP TIME_WAIT timeout - closing connection\n");
    // Close connection
}

tcp_timers_t* create_tcp_timers() {
    tcp_timers_t *timers = malloc(sizeof(tcp_timers_t));
    timers->wheel = create_timer_wheel(1000, 10); // 10ms resolution
    
    // Set up standard TCP timers
    timers->retransmit_timer = add_timer(timers->wheel, 200, tcp_retransmit_timeout, NULL);
    timers->keepalive_timer = add_timer(timers->wheel, 7200000, tcp_keepalive_timeout, NULL); // 2 hours
    timers->time_wait_timer = add_timer(timers->wheel, 240000, tcp_time_wait_timeout, NULL); // 4 minutes
    
    return timers;
}
```

### High-Performance Socket Programming

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <errno.h>

// Event-driven server using epoll for scalability
typedef struct {
    int epoll_fd;
    struct epoll_event *events;
    int max_events;
    int server_fd;
} epoll_server_t;

// Make socket non-blocking
int make_socket_non_blocking(int sockfd) {
    int flags = fcntl(sockfd, F_GETFL, 0);
    if (flags == -1) {
        perror("fcntl F_GETFL");
        return -1;
    }
    
    flags |= O_NONBLOCK;
    if (fcntl(sockfd, F_SETFL, flags) == -1) {
        perror("fcntl F_SETFL");
        return -1;
    }
    
    return 0;
}

// Configure socket for high performance
void configure_high_performance_socket(int sockfd) {
    int opt = 1;
    
    // Enable address reuse
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
    
    // Increase buffer sizes
    int buffer_size = 1024 * 1024; // 1MB
    setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &buffer_size, sizeof(buffer_size));
    setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &buffer_size, sizeof(buffer_size));
    
    // Disable Nagle's algorithm for low latency
    setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    
    // Enable keep-alive
    setsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE, &opt, sizeof(opt));
    
    // Set TCP keep-alive parameters
    int keepidle = 600;   // 10 minutes
    int keepintvl = 60;   // 1 minute
    int keepcnt = 3;      // 3 probes
    
    setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPIDLE, &keepidle, sizeof(keepidle));
    setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPINTVL, &keepintvl, sizeof(keepintvl));
    setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPCNT, &keepcnt, sizeof(keepcnt));
    
    printf("High-performance socket configuration applied\n");
}

// Create high-performance epoll server
epoll_server_t* create_epoll_server(int port, int max_connections) {
    epoll_server_t *server = malloc(sizeof(epoll_server_t));
    
    // Create epoll instance
    server->epoll_fd = epoll_create1(EPOLL_CLOEXEC);
    if (server->epoll_fd == -1) {
        perror("epoll_create1");
        free(server);
        return NULL;
    }
    
    // Create server socket
    server->server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server->server_fd == -1) {
        perror("socket");
        close(server->epoll_fd);
        free(server);
        return NULL;
    }
    
    // Configure server socket
    configure_high_performance_socket(server->server_fd);
    make_socket_non_blocking(server->server_fd);
    
    // Bind and listen
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);
    
    if (bind(server->server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
        perror("bind");
        close(server->server_fd);
        close(server->epoll_fd);
        free(server);
        return NULL;
    }
    
    if (listen(server->server_fd, SOMAXCONN) == -1) {
        perror("listen");
        close(server->server_fd);
        close(server->epoll_fd);
        free(server);
        return NULL;
    }
    
    // Add server socket to epoll
    struct epoll_event event;
    event.events = EPOLLIN | EPOLLET; // Edge-triggered
    event.data.fd = server->server_fd;
    
    if (epoll_ctl(server->epoll_fd, EPOLL_CTL_ADD, server->server_fd, &event) == -1) {
        perror("epoll_ctl");
        close(server->server_fd);
        close(server->epoll_fd);
        free(server);
        return NULL;
    }
    
    server->max_events = max_connections;
    server->events = malloc(max_connections * sizeof(struct epoll_event));
    
    printf("High-performance epoll server created on port %d\n", port);
    return server;
}

// Handle new client connections
void handle_new_connection(epoll_server_t *server) {
    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        int client_fd = accept(server->server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // No more connections to accept
                break;
            } else {
                perror("accept");
                break;
            }
        }
        
        // Configure client socket
        configure_high_performance_socket(client_fd);
        make_socket_non_blocking(client_fd);
        
        // Add client to epoll
        struct epoll_event event;
        event.events = EPOLLIN | EPOLLET; // Edge-triggered
        event.data.fd = client_fd;
        
        if (epoll_ctl(server->epoll_fd, EPOLL_CTL_ADD, client_fd, &event) == -1) {
            perror("epoll_ctl");
            close(client_fd);
            continue;
        }
        
        printf("New client connected: %s:%d (fd=%d)\n",
               inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port), client_fd);
    }
}

// Handle client data with zero-copy techniques
void handle_client_data(int client_fd) {
    char buffer[8192];
    ssize_t bytes_read;
    
    while (1) {
        bytes_read = recv(client_fd, buffer, sizeof(buffer), 0);
        
        if (bytes_read == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // No more data to read
                break;
            } else {
                perror("recv");
                break;
            }
        } else if (bytes_read == 0) {
            // Client disconnected
            printf("Client disconnected (fd=%d)\n", client_fd);
            close(client_fd);
            break;
        } else {
            // Echo data back (simple example)
            ssize_t bytes_sent = 0;
            ssize_t total_sent = 0;
            
            while (total_sent < bytes_read) {
                bytes_sent = send(client_fd, buffer + total_sent, 
                                bytes_read - total_sent, MSG_NOSIGNAL);
                
                if (bytes_sent == -1) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) {
                        // Socket buffer full, would need to use EPOLLOUT
                        break;
                    } else {
                        perror("send");
                        close(client_fd);
                        return;
                    }
                }
                
                total_sent += bytes_sent;
            }
        }
    }
}

// Main server event loop
void run_epoll_server(epoll_server_t *server) {
    printf("Starting high-performance server event loop...\n");
    
    while (1) {
        int nfds = epoll_wait(server->epoll_fd, server->events, server->max_events, -1);
        
        if (nfds == -1) {
            perror("epoll_wait");
            break;
        }
        
        for (int i = 0; i < nfds; i++) {
            int fd = server->events[i].data.fd;
            
            if (fd == server->server_fd) {
                // New connection
                handle_new_connection(server);
            } else {
                // Client data
                if (server->events[i].events & EPOLLIN) {
                    handle_client_data(fd);
                }
                
                if (server->events[i].events & (EPOLLHUP | EPOLLERR)) {
                    printf("Client error or hangup (fd=%d)\n", fd);
                    close(fd);
                }
            }
        }
    }
}
```

### Zero-Copy and Memory-Mapped I/O

```c
#include <sys/mman.h>
#include <sys/sendfile.h>
#include <fcntl.h>

// Memory-mapped file transfer
typedef struct {
    void *mapped_memory;
    size_t file_size;
    int fd;
} mmap_file_t;

// Map file into memory for zero-copy operations
mmap_file_t* mmap_file_open(const char *filename) {
    mmap_file_t *mf = malloc(sizeof(mmap_file_t));
    
    mf->fd = open(filename, O_RDONLY);
    if (mf->fd == -1) {
        perror("open");
        free(mf);
        return NULL;
    }
    
    // Get file size
    struct stat st;
    if (fstat(mf->fd, &st) == -1) {
        perror("fstat");
        close(mf->fd);
        free(mf);
        return NULL;
    }
    
    mf->file_size = st.st_size;
    
    // Map file into memory
    mf->mapped_memory = mmap(NULL, mf->file_size, PROT_READ, MAP_PRIVATE, mf->fd, 0);
    if (mf->mapped_memory == MAP_FAILED) {
        perror("mmap");
        close(mf->fd);
        free(mf);
        return NULL;
    }
    
    // Advise kernel about access pattern
    madvise(mf->mapped_memory, mf->file_size, MADV_SEQUENTIAL);
    
    printf("Memory-mapped file: %s (%zu bytes)\n", filename, mf->file_size);
    return mf;
}

// Send memory-mapped file using sendfile (zero-copy)
ssize_t send_mmap_file(int socket_fd, mmap_file_t *mf, off_t offset, size_t count) {
    return sendfile(socket_fd, mf->fd, &offset, count);
}

// Batch processing for improved throughput
typedef struct {
    struct iovec *iov;
    int count;
    int capacity;
} batch_buffer_t;

batch_buffer_t* create_batch_buffer(int capacity) {
    batch_buffer_t *batch = malloc(sizeof(batch_buffer_t));
    batch->iov = malloc(capacity * sizeof(struct iovec));
    batch->count = 0;
    batch->capacity = capacity;
    return batch;
}

void batch_add_buffer(batch_buffer_t *batch, void *data, size_t len) {
    if (batch->count < batch->capacity) {
        batch->iov[batch->count].iov_base = data;
        batch->iov[batch->count].iov_len = len;
        batch->count++;
    }
}

ssize_t batch_send(int sockfd, batch_buffer_t *batch) {
    ssize_t total_sent = writev(sockfd, batch->iov, batch->count);
    batch->count = 0; // Reset batch
    return total_sent;
}
```

### Network Latency Optimization

```c
#include <time.h>
#include <sched.h>

// High-resolution timing for latency measurement
typedef struct {
    struct timespec start;
    struct timespec end;
} latency_timer_t;

void latency_timer_start(latency_timer_t *timer) {
    clock_gettime(CLOCK_MONOTONIC, &timer->start);
}

double latency_timer_end(latency_timer_t *timer) {
    clock_gettime(CLOCK_MONOTONIC, &timer->end);
    
    double elapsed = (timer->end.tv_sec - timer->start.tv_sec) * 1000000.0;
    elapsed += (timer->end.tv_nsec - timer->start.tv_nsec) / 1000.0;
    
    return elapsed; // microseconds
}

// CPU affinity for consistent performance
void set_cpu_affinity(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) == -1) {
        perror("sched_setaffinity");
    } else {
        printf("Thread bound to CPU %d\n", cpu_id);
    }
}

// Real-time scheduling for low-latency applications
void set_realtime_priority(int priority) {
    struct sched_param param;
    param.sched_priority = priority;
    
    if (sched_setscheduler(0, SCHED_FIFO, &param) == -1) {
        perror("sched_setscheduler");
    } else {
        printf("Real-time priority set to %d\n", priority);
    }
}

// Network performance monitoring
typedef struct {
    uint64_t packets_sent;
    uint64_t packets_received;
    uint64_t bytes_sent;
    uint64_t bytes_received;
    uint64_t errors;
    uint64_t retransmissions;
    double avg_latency;
    double max_latency;
    double min_latency;
} network_stats_t;

void update_network_stats(network_stats_t *stats, size_t bytes, double latency) {
    stats->bytes_sent += bytes;
    stats->packets_sent++;
    
    // Update latency statistics
    if (stats->packets_sent == 1) {
        stats->avg_latency = latency;
        stats->max_latency = latency;
        stats->min_latency = latency;
    } else {
        stats->avg_latency = (stats->avg_latency * (stats->packets_sent - 1) + latency) / stats->packets_sent;
        if (latency > stats->max_latency) stats->max_latency = latency;
        if (latency < stats->min_latency) stats->min_latency = latency;
    }
}

void print_network_stats(network_stats_t *stats) {
    printf("Network Statistics:\n");
    printf("  Packets sent: %lu\n", stats->packets_sent);
    printf("  Bytes sent: %lu\n", stats->bytes_sent);
    printf("  Avg latency: %.2f μs\n", stats->avg_latency);
    printf("  Min latency: %.2f μs\n", stats->min_latency);
    printf("  Max latency: %.2f μs\n", stats->max_latency);
    printf("  Errors: %lu\n", stats->errors);
}
```

### Performance Testing and Benchmarking

```c
// Network throughput test
void throughput_test(const char *server_ip, int port, size_t data_size, int duration_sec) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    configure_high_performance_socket(sockfd);
    
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    inet_pton(AF_INET, server_ip, &server_addr.sin_addr);
    
    if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
        perror("connect");
        close(sockfd);
        return;
    }
    
    char *test_data = malloc(data_size);
    memset(test_data, 'A', data_size);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    uint64_t total_bytes = 0;
    uint64_t total_packets = 0;
    
    while (1) {
        struct timespec current;
        clock_gettime(CLOCK_MONOTONIC, &current);
        
        double elapsed = (current.tv_sec - start.tv_sec) + 
                        (current.tv_nsec - start.tv_nsec) / 1000000000.0;
        
        if (elapsed >= duration_sec) break;
        
        ssize_t sent = send(sockfd, test_data, data_size, 0);
        if (sent > 0) {
            total_bytes += sent;
            total_packets++;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double total_time = (end.tv_sec - start.tv_sec) + 
                       (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    double throughput_mbps = (total_bytes * 8.0) / (total_time * 1000000.0);
    double packet_rate = total_packets / total_time;
    
    printf("Throughput Test Results:\n");
    printf("  Duration: %.2f seconds\n", total_time);
    printf("  Total bytes: %lu\n", total_bytes);
    printf("  Total packets: %lu\n", total_packets);
    printf("  Throughput: %.2f Mbps\n", throughput_mbps);
    printf("  Packet rate: %.2f packets/sec\n", packet_rate);
    
    free(test_data);
    close(sockfd);
}

// Example usage of performance optimization
void performance_optimization_example() {
    printf("=== Network Performance Optimization Example ===\n");
    
    // Set CPU affinity and real-time priority for consistent performance
    set_cpu_affinity(0);  // Bind to CPU 0
    // set_realtime_priority(50);  // Uncomment for real-time scheduling
    
    // Create high-performance server
    epoll_server_t *server = create_epoll_server(8080, 10000);
    if (!server) return;
    
    printf("Server ready for high-performance connections\n");
    printf("Use: telnet localhost 8080 to test\n");
    
    // Run throughput test (in separate process/thread)
    // throughput_test("127.0.0.1", 8080, 1024, 10);
    
    // Start server event loop
    // run_epoll_server(server);
}
```

## Learning Assessment

### Practical Projects

**Project 1: Multi-threaded TCP Server**
- Implement a thread-per-connection server
- Add proper synchronization and resource management
- Compare performance with event-driven approach

**Project 2: Custom Protocol Implementation**
- Design a simple reliable protocol over UDP
- Implement acknowledgments, retransmissions, and flow control
- Create protocol control blocks for state management

**Project 3: High-Performance Packet Processor**
- Build a packet sniffer using raw sockets
- Implement efficient buffer management
- Add packet filtering and analysis capabilities

**Project 4: Network Benchmarking Tool**
- Create a tool to measure network throughput and latency
- Implement various optimization techniques
- Compare different socket configurations

### Study Materials

**Essential Reading:**
- "Unix Network Programming" by W. Richard Stevens (Volumes 1 & 2)
- "TCP/IP Illustrated" by W. Richard Stevens (Volume 1)
- "High Performance Browser Networking" by Ilya Grigorik
- "The Linux Programming Interface" by Michael Kerrisk (Chapters 56-61)

**Advanced Topics:**
- DPDK (Data Plane Development Kit) documentation
- Linux kernel networking source code
- RFC 793 (TCP), RFC 768 (UDP), RFC 791 (IP)

**Tools and Debugging:**
- Wireshark for packet analysis
- `tcpdump` for command-line packet capture
- `netstat`, `ss`, `lsof` for connection monitoring
- `perf` for performance profiling
- `strace` for system call tracing

### Development Environment

**Required Tools:**
```bash
# Development tools
sudo apt-get install build-essential
sudo apt-get install libpcap-dev  # For packet capture
sudo apt-get install wireshark tshark tcpdump

# Performance tools
sudo apt-get install linux-tools-generic  # perf
sudo apt-get install sysstat iotop htop

# Network testing
sudo apt-get install netperf iperf3 nmap
```

**Compilation Examples:**
```bash
# Basic compilation
gcc -o tcp_server tcp_server.c -lpthread

# With debugging and optimization
gcc -g -O2 -Wall -Wextra -o tcp_server tcp_server.c -lpthread

# For raw socket programs (requires root)
gcc -o packet_sniffer packet_sniffer.c -lpcap
sudo ./packet_sniffer

# Link with additional libraries
gcc -o network_app network_app.c -lpthread -lrt -lm
```

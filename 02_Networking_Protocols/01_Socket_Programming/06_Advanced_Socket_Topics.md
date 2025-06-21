# Advanced Socket Topics

*Last Updated: June 21, 2025*

## Overview

This module covers advanced socket programming topics including socket options, out-of-band data, Unix domain sockets, and raw sockets. These topics are essential for building sophisticated network applications with fine-tuned performance and specialized functionality.

## Learning Objectives

By the end of this module, you should be able to:
- Configure socket options for performance optimization
- Handle out-of-band data in TCP connections
- Use Unix domain sockets for local IPC
- Work with raw sockets for low-level networking
- Implement custom protocols and packet analysis

## Topics Covered

### 1. Socket Options
- Performance tuning options
- Buffer size configuration
- Keep-alive settings
- Nagle's algorithm and TCP_NODELAY
- Socket behavior modification

### 2. Out-of-Band Data
- Understanding urgent data in TCP
- Sending and receiving OOB data
- Implementation patterns and usage scenarios

### 3. Unix Domain Sockets
- Local inter-process communication
- Datagram vs stream Unix sockets
- Performance advantages over TCP sockets

### 4. Raw Sockets
- Direct protocol access
- Custom protocol implementation
- Packet sniffing and injection
- Network analysis tools

## Practical Exercises

1. **Socket Option Benchmarking**
   - Compare performance with different socket options
   - Measure throughput and latency impacts

2. **Out-of-Band Data Handler**
   - Implement urgent data processing
   - Create interrupt-style communication

3. **Unix Domain Socket IPC**
   - Replace TCP sockets with Unix sockets
   - Benchmark performance improvements

4. **Network Packet Analyzer**
   - Build a simple packet capture tool
   - Analyze different protocol headers

## Code Examples

### 1. Socket Options Configuration

```c
#include <sys/socket.h>
#include <netinet/tcp.h>

int configure_socket_options(int sockfd) {
    int option_value;
    socklen_t option_len = sizeof(option_value);
    
    // 1. Disable Nagle's algorithm for low latency
    option_value = 1;
    if (setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, 
                   &option_value, sizeof(option_value)) < 0) {
        perror("TCP_NODELAY failed");
        return -1;
    }
    
    // 2. Enable keep-alive
    option_value = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE, 
                   &option_value, sizeof(option_value)) < 0) {
        perror("SO_KEEPALIVE failed");
        return -1;
    }
    
    // 3. Set keep-alive parameters (Linux-specific)
    #ifdef TCP_KEEPIDLE
    option_value = 600;  // Start probing after 10 minutes
    setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPIDLE, &option_value, sizeof(option_value));
    
    option_value = 60;   // Probe every 60 seconds
    setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPINTVL, &option_value, sizeof(option_value));
    
    option_value = 3;    // Drop after 3 failed probes
    setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPCNT, &option_value, sizeof(option_value));
    #endif
    
    // 4. Set socket buffer sizes
    option_value = 65536;  // 64KB
    if (setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, 
                   &option_value, sizeof(option_value)) < 0) {
        perror("SO_SNDBUF failed");
    }
    
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, 
                   &option_value, sizeof(option_value)) < 0) {
        perror("SO_RCVBUF failed");
    }
    
    // 5. Enable address reuse
    option_value = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, 
                   &option_value, sizeof(option_value)) < 0) {
        perror("SO_REUSEADDR failed");
    }
    
    // 6. Set socket to non-blocking mode
    int flags = fcntl(sockfd, F_GETFL, 0);
    if (fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) < 0) {
        perror("Setting non-blocking failed");
        return -1;
    }
    
    return 0;
}

void print_socket_options(int sockfd) {
    int value;
    socklen_t len = sizeof(value);
    
    printf("Socket Options for fd %d:\n", sockfd);
    
    if (getsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &value, &len) == 0) {
        printf("  Send buffer size: %d bytes\n", value);
    }
    
    if (getsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &value, &len) == 0) {
        printf("  Receive buffer size: %d bytes\n", value);
    }
    
    if (getsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE, &value, &len) == 0) {
        printf("  Keep-alive: %s\n", value ? "enabled" : "disabled");
    }
    
    if (getsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &value, &len) == 0) {
        printf("  TCP_NODELAY: %s\n", value ? "enabled" : "disabled");
    }
}
```

### 2. Out-of-Band Data Handling

```c
#include <signal.h>

volatile int oob_received = 0;

void sigurg_handler(int sig) {
    oob_received = 1;
}

int setup_oob_handling(int sockfd) {
    // Set up signal handler for urgent data
    signal(SIGURG, sigurg_handler);
    
    // Tell kernel to send SIGURG to this process
    if (fcntl(sockfd, F_SETOWN, getpid()) < 0) {
        perror("fcntl F_SETOWN");
        return -1;
    }
    
    return 0;
}

int send_oob_data(int sockfd, const char* data) {
    printf("Sending OOB data: %s\n", data);
    return send(sockfd, data, strlen(data), MSG_OOB);
}

int receive_oob_data(int sockfd) {
    char oob_buffer[1024];
    
    if (oob_received) {
        ssize_t bytes = recv(sockfd, oob_buffer, sizeof(oob_buffer) - 1, MSG_OOB);
        if (bytes > 0) {
            oob_buffer[bytes] = '\0';
            printf("Received OOB data: %s\n", oob_buffer);
            oob_received = 0;
            return bytes;
        }
    }
    
    return 0;
}

void oob_server_example(int server_fd) {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    
    int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
    if (client_fd < 0) {
        perror("Accept failed");
        return;
    }
    
    setup_oob_handling(client_fd);
    
    char buffer[1024];
    while (1) {
        // Check for out-of-band data
        receive_oob_data(client_fd);
        
        // Regular data handling
        fd_set readfds;
        struct timeval timeout = {1, 0};  // 1 second timeout
        
        FD_ZERO(&readfds);
        FD_SET(client_fd, &readfds);
        
        int result = select(client_fd + 1, &readfds, NULL, NULL, &timeout);
        if (result > 0 && FD_ISSET(client_fd, &readfds)) {
            ssize_t bytes = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
            if (bytes <= 0) break;
            
            buffer[bytes] = '\0';
            printf("Received normal data: %s\n", buffer);
            
            // Echo back
            send(client_fd, buffer, bytes, 0);
        }
    }
    
    close(client_fd);
}
```

### 3. Unix Domain Sockets

```c
#include <sys/un.h>

#define SOCKET_PATH "/tmp/my_socket"

int create_unix_server(const char* path) {
    int sockfd;
    struct sockaddr_un server_addr;
    
    // Create Unix domain socket
    sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("Unix socket creation failed");
        return -1;
    }
    
    // Remove existing socket file
    unlink(path);
    
    // Configure server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sun_family = AF_UNIX;
    strncpy(server_addr.sun_path, path, sizeof(server_addr.sun_path) - 1);
    
    // Bind socket
    if (bind(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Unix socket bind failed");
        close(sockfd);
        return -1;
    }
    
    // Listen for connections
    if (listen(sockfd, 5) < 0) {
        perror("Unix socket listen failed");
        close(sockfd);
        unlink(path);
        return -1;
    }
    
    printf("Unix domain server listening on %s\n", path);
    return sockfd;
}

int connect_unix_client(const char* path) {
    int sockfd;
    struct sockaddr_un server_addr;
    
    sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("Unix socket creation failed");
        return -1;
    }
    
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sun_family = AF_UNIX;
    strncpy(server_addr.sun_path, path, sizeof(server_addr.sun_path) - 1);
    
    if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Unix socket connect failed");
        close(sockfd);
        return -1;
    }
    
    return sockfd;
}

// Unix domain datagram sockets
int create_unix_dgram_socket(const char* path) {
    int sockfd;
    struct sockaddr_un addr;
    
    sockfd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Unix datagram socket creation failed");
        return -1;
    }
    
    if (path) {
        unlink(path);
        
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
        
        if (bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("Unix datagram socket bind failed");
            close(sockfd);
            return -1;
        }
    }
    
    return sockfd;
}

void unix_ipc_example() {
    pid_t pid = fork();
    
    if (pid == 0) {
        // Child process - client
        sleep(1);  // Let parent set up server
        
        int client_fd = connect_unix_client(SOCKET_PATH);
        if (client_fd >= 0) {
            const char* message = "Hello from Unix domain socket!";
            send(client_fd, message, strlen(message), 0);
            
            char response[1024];
            ssize_t bytes = recv(client_fd, response, sizeof(response) - 1, 0);
            if (bytes > 0) {
                response[bytes] = '\0';
                printf("Client received: %s\n", response);
            }
            
            close(client_fd);
        }
        exit(0);
    } else if (pid > 0) {
        // Parent process - server
        int server_fd = create_unix_server(SOCKET_PATH);
        if (server_fd >= 0) {
            struct sockaddr_un client_addr;
            socklen_t client_len = sizeof(client_addr);
            
            int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
            if (client_fd >= 0) {
                char buffer[1024];
                ssize_t bytes = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
                if (bytes > 0) {
                    buffer[bytes] = '\0';
                    printf("Server received: %s\n", buffer);
                    
                    // Echo back
                    send(client_fd, buffer, bytes, 0);
                }
                close(client_fd);
            }
            
            close(server_fd);
            unlink(SOCKET_PATH);
        }
        
        wait(NULL);  // Wait for child
    }
}
```

### 4. Raw Sockets

```c
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <netinet/ip_icmp.h>

// Note: Raw sockets typically require root privileges

int create_raw_socket(int protocol) {
    int sockfd = socket(AF_INET, SOCK_RAW, protocol);
    if (sockfd < 0) {
        perror("Raw socket creation failed (need root privileges)");
        return -1;
    }
    
    // Include IP headers in received packets
    int on = 1;
    if (setsockopt(sockfd, IPPROTO_IP, IP_HDRINCL, &on, sizeof(on)) < 0) {
        perror("setsockopt IP_HDRINCL");
        close(sockfd);
        return -1;
    }
    
    return sockfd;
}

// Simple IP header structure
struct ip_header {
    unsigned char  version_ihl;     // Version (4 bits) + IHL (4 bits)
    unsigned char  type_of_service; // Type of service
    unsigned short total_length;    // Total length
    unsigned short identification;  // Identification
    unsigned short flags_fragment;  // Flags (3 bits) + Fragment offset (13 bits)
    unsigned char  time_to_live;    // Time to live
    unsigned char  protocol;        // Protocol
    unsigned short header_checksum; // Header checksum
    unsigned int   source_address;  // Source address
    unsigned int   dest_address;    // Destination address
};

void print_ip_header(struct ip_header* ip_hdr) {
    printf("IP Header:\n");
    printf("  Version: %d\n", (ip_hdr->version_ihl >> 4) & 0x0F);
    printf("  Header Length: %d bytes\n", (ip_hdr->version_ihl & 0x0F) * 4);
    printf("  Type of Service: 0x%02x\n", ip_hdr->type_of_service);
    printf("  Total Length: %d\n", ntohs(ip_hdr->total_length));
    printf("  Identification: 0x%04x\n", ntohs(ip_hdr->identification));
    printf("  TTL: %d\n", ip_hdr->time_to_live);
    printf("  Protocol: %d\n", ip_hdr->protocol);
    printf("  Header Checksum: 0x%04x\n", ntohs(ip_hdr->header_checksum));
    
    struct in_addr addr;
    addr.s_addr = ip_hdr->source_address;
    printf("  Source: %s\n", inet_ntoa(addr));
    
    addr.s_addr = ip_hdr->dest_address;
    printf("  Destination: %s\n", inet_ntoa(addr));
}

void packet_sniffer(int protocol) {
    int raw_sock = create_raw_socket(protocol);
    if (raw_sock < 0) return;
    
    printf("Starting packet capture for protocol %d...\n", protocol);
    printf("Press Ctrl+C to stop\n\n");
    
    char buffer[65536];
    struct sockaddr_in source_addr;
    socklen_t addr_len = sizeof(source_addr);
    
    while (1) {
        ssize_t packet_size = recvfrom(raw_sock, buffer, sizeof(buffer), 0,
                                      (struct sockaddr*)&source_addr, &addr_len);
        
        if (packet_size < 0) {
            perror("Packet receive failed");
            break;
        }
        
        printf("Captured packet (%zd bytes):\n", packet_size);
        
        // Parse IP header
        struct ip_header* ip_hdr = (struct ip_header*)buffer;
        print_ip_header(ip_hdr);
        
        // Parse payload based on protocol
        int ip_header_len = (ip_hdr->version_ihl & 0x0F) * 4;
        char* payload = buffer + ip_header_len;
        int payload_len = packet_size - ip_header_len;
        
        switch (ip_hdr->protocol) {
            case IPPROTO_TCP: {
                struct tcphdr* tcp_hdr = (struct tcphdr*)payload;
                printf("  TCP: %d -> %d\n", 
                       ntohs(tcp_hdr->source), ntohs(tcp_hdr->dest));
                break;
            }
            case IPPROTO_UDP: {
                struct udphdr* udp_hdr = (struct udphdr*)payload;
                printf("  UDP: %d -> %d\n", 
                       ntohs(udp_hdr->source), ntohs(udp_hdr->dest));
                break;
            }
            case IPPROTO_ICMP: {
                struct icmphdr* icmp_hdr = (struct icmphdr*)payload;
                printf("  ICMP: type=%d, code=%d\n", 
                       icmp_hdr->type, icmp_hdr->code);
                break;
            }
        }
        
        printf("\n");
    }
    
    close(raw_sock);
}

// Simple ping implementation using raw sockets
unsigned short checksum(void* b, int len) {
    unsigned short* buf = b;
    unsigned int sum = 0;
    unsigned short result;
    
    while (len > 1) {
        sum += *buf++;
        len -= 2;
    }
    
    if (len == 1) {
        sum += *(unsigned char*)buf << 8;
    }
    
    while (sum >> 16) {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    
    result = ~sum;
    return result;
}

int send_ping(const char* target_ip) {
    int raw_sock = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP);
    if (raw_sock < 0) {
        perror("Raw socket creation failed (need root privileges)");
        return -1;
    }
    
    struct sockaddr_in target_addr;
    memset(&target_addr, 0, sizeof(target_addr));
    target_addr.sin_family = AF_INET;
    target_addr.sin_addr.s_addr = inet_addr(target_ip);
    
    // Create ICMP packet
    struct icmphdr icmp_hdr;
    memset(&icmp_hdr, 0, sizeof(icmp_hdr));
    icmp_hdr.type = ICMP_ECHO;
    icmp_hdr.code = 0;
    icmp_hdr.un.echo.id = getpid();
    icmp_hdr.un.echo.sequence = 1;
    icmp_hdr.checksum = 0;
    icmp_hdr.checksum = checksum(&icmp_hdr, sizeof(icmp_hdr));
    
    // Send ping
    if (sendto(raw_sock, &icmp_hdr, sizeof(icmp_hdr), 0,
              (struct sockaddr*)&target_addr, sizeof(target_addr)) < 0) {
        perror("Ping send failed");
        close(raw_sock);
        return -1;
    }
    
    printf("Ping sent to %s\n", target_ip);
    
    // Wait for reply (simplified - should use proper timeout)
    char reply_buffer[1024];
    struct sockaddr_in reply_addr;
    socklen_t reply_len = sizeof(reply_addr);
    
    if (recvfrom(raw_sock, reply_buffer, sizeof(reply_buffer), 0,
                (struct sockaddr*)&reply_addr, &reply_len) > 0) {
        printf("Ping reply received from %s\n", inet_ntoa(reply_addr.sin_addr));
    }
    
    close(raw_sock);
    return 0;
}
```

## Socket Options Reference

### Performance-Related Options

| Option | Level | Description | Use Case |
|--------|-------|-------------|----------|
| TCP_NODELAY | IPPROTO_TCP | Disable Nagle's algorithm | Low-latency applications |
| SO_SNDBUF/SO_RCVBUF | SOL_SOCKET | Buffer sizes | High-throughput applications |
| SO_KEEPALIVE | SOL_SOCKET | Enable keep-alive probes | Long-lived connections |
| TCP_KEEPIDLE | IPPROTO_TCP | Keep-alive idle time | Fine-tune keep-alive |
| SO_REUSEADDR | SOL_SOCKET | Reuse local addresses | Server restart scenarios |

### Advanced Options

| Option | Level | Description | Use Case |
|--------|-------|-------------|----------|
| SO_BROADCAST | SOL_SOCKET | Enable broadcast | UDP broadcast applications |
| IP_MULTICAST_TTL | IPPROTO_IP | Multicast TTL | Multicast scope control |
| SO_TIMESTAMP | SOL_SOCKET | Timestamp packets | Network timing analysis |
| TCP_CONGESTION | IPPROTO_TCP | Congestion control algorithm | Performance tuning |

## Best Practices

### Socket Options
1. **Profile before optimizing** - measure actual performance impact
2. **Consider application requirements** - latency vs throughput
3. **Test on target platforms** - options vary by OS
4. **Monitor resource usage** - larger buffers use more memory

### Unix Domain Sockets
1. **Use for local IPC** - much faster than TCP for local communication
2. **Clean up socket files** - remove on exit or startup
3. **Set proper permissions** - secure socket files appropriately
4. **Consider abstract namespace** - Linux-specific feature

### Raw Sockets
1. **Require root privileges** - handle gracefully
2. **Validate packet data** - raw data can be malformed
3. **Handle endianness** - network byte order conversion
4. **Be careful with packet injection** - can affect network

## Assessment Checklist

- [ ] Can configure socket options for different scenarios
- [ ] Understands and implements out-of-band data handling
- [ ] Successfully uses Unix domain sockets for IPC
- [ ] Works with raw sockets for packet analysis
- [ ] Implements custom network protocols
- [ ] Optimizes socket performance for specific use cases

## Next Steps

After mastering advanced socket topics:
- Explore secure socket programming with SSL/TLS
- Study high-performance networking libraries
- Learn about kernel bypass techniques (DPDK, netmap)

## Resources

- "UNIX Network Programming, Volume 1" by W. Richard Stevens (Chapters 7, 14, 15)
- Linux man pages: socket(7), tcp(7), udp(7), unix(7), raw(7)
- "TCP/IP Illustrated, Volume 1" by W. Richard Stevens
- Kernel documentation for socket options

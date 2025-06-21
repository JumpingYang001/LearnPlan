# UDP Socket Programming

*Last Updated: June 21, 2025*

## Overview

This module covers UDP (User Datagram Protocol) socket programming, focusing on connectionless communication patterns. UDP provides a lightweight, fast communication mechanism without the overhead of connection establishment and reliability guarantees of TCP.

## Learning Objectives

By the end of this module, you should be able to:
- Create UDP sockets for both client and server applications
- Send and receive datagrams effectively
- Understand connectionless communication patterns
- Handle packet loss and reordering scenarios
- Implement reliability mechanisms over UDP when needed

## Topics Covered

### Creating UDP Sockets
- UDP socket creation with `SOCK_DGRAM`
- Socket configuration for UDP
- Differences from TCP socket setup

### Sending and Receiving Datagrams
- `sendto()` and `recvfrom()` functions
- Address specification for each packet
- Datagram size limitations
- Buffer management for UDP

### Connectionless Communication Patterns
- Stateless server design
- Client identification strategies
- Request-response patterns
- Broadcast and multicast communication

### Handling Packet Loss and Reordering
- Understanding UDP's unreliable nature
- Detecting packet loss
- Handling out-of-order packets
- Duplicate packet detection

### Implementing Reliability over UDP
- Acknowledgment mechanisms
- Timeout and retransmission
- Sequence numbering
- Flow control basics

## Practical Exercises

1. **Simple UDP Echo Server/Client**
   - Basic UDP communication
   - Single request-response pattern

2. **Reliable UDP File Transfer**
   - Implement acknowledgments
   - Handle packet loss and retransmission

3. **UDP Chat Application**
   - Multi-client chat server
   - Broadcast message handling

4. **Network Time Protocol (SNTP) Client**
   - Implement a simple SNTP client
   - Handle time synchronization

## Code Examples

### Basic UDP Server
```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>

int create_udp_server(int port) {
    int sockfd;
    struct sockaddr_in server_addr;
    
    // Create UDP socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("UDP socket creation failed");
        return -1;
    }
    
    // Configure server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);
    
    // Bind socket
    if (bind(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("UDP bind failed");
        close(sockfd);
        return -1;
    }
    
    printf("UDP server listening on port %d\n", port);
    return sockfd;
}

void udp_echo_server(int sockfd) {
    char buffer[1024];
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    ssize_t bytes_received;
    
    while (1) {
        // Receive datagram
        bytes_received = recvfrom(sockfd, buffer, sizeof(buffer) - 1, 0,
                                 (struct sockaddr*)&client_addr, &client_len);
        
        if (bytes_received < 0) {
            perror("Receive failed");
            continue;
        }
        
        buffer[bytes_received] = '\0';
        printf("Received from %s:%d: %s\n",
               inet_ntoa(client_addr.sin_addr),
               ntohs(client_addr.sin_port),
               buffer);
        
        // Echo back to client
        if (sendto(sockfd, buffer, bytes_received, 0,
                  (struct sockaddr*)&client_addr, client_len) < 0) {
            perror("Send failed");
        }
    }
}
```

### Basic UDP Client
```c
int create_udp_client() {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("UDP socket creation failed");
        return -1;
    }
    return sockfd;
}

void udp_client_communication(int sockfd, const char* server_ip, int port) {
    struct sockaddr_in server_addr;
    socklen_t server_len = sizeof(server_addr);
    char send_buffer[1024];
    char recv_buffer[1024];
    
    // Configure server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    server_addr.sin_addr.s_addr = inet_addr(server_ip);
    
    while (1) {
        printf("Enter message (or 'quit' to exit): ");
        if (!fgets(send_buffer, sizeof(send_buffer), stdin)) {
            break;
        }
        
        if (strncmp(send_buffer, "quit", 4) == 0) {
            break;
        }
        
        // Send message
        ssize_t sent = sendto(sockfd, send_buffer, strlen(send_buffer), 0,
                             (struct sockaddr*)&server_addr, server_len);
        if (sent < 0) {
            perror("Send failed");
            continue;
        }
        
        // Receive response
        ssize_t received = recvfrom(sockfd, recv_buffer, sizeof(recv_buffer) - 1, 0,
                                   (struct sockaddr*)&server_addr, &server_len);
        if (received < 0) {
            perror("Receive failed");
            continue;
        }
        
        recv_buffer[received] = '\0';
        printf("Server response: %s", recv_buffer);
    }
}
```

### Reliable UDP with Acknowledgments
```c
#include <sys/time.h>
#include <errno.h>

#define MAX_RETRIES 3
#define TIMEOUT_SEC 2

typedef struct {
    uint32_t seq_num;
    uint32_t ack_num;
    uint16_t flags;
    uint16_t data_len;
    char data[];
} udp_packet_t;

#define FLAG_ACK 0x01
#define FLAG_SYN 0x02
#define FLAG_FIN 0x04

int send_reliable_udp(int sockfd, const struct sockaddr* dest_addr, 
                     socklen_t addrlen, const void* data, size_t len, 
                     uint32_t seq_num) {
    udp_packet_t* packet = malloc(sizeof(udp_packet_t) + len);
    packet->seq_num = htonl(seq_num);
    packet->ack_num = 0;
    packet->flags = 0;
    packet->data_len = htons(len);
    memcpy(packet->data, data, len);
    
    struct timeval timeout;
    fd_set readfds;
    int retries = 0;
    
    while (retries < MAX_RETRIES) {
        // Send packet
        if (sendto(sockfd, packet, sizeof(udp_packet_t) + len, 0, 
                  dest_addr, addrlen) < 0) {
            perror("Send failed");
            free(packet);
            return -1;
        }
        
        // Wait for ACK
        FD_ZERO(&readfds);
        FD_SET(sockfd, &readfds);
        timeout.tv_sec = TIMEOUT_SEC;
        timeout.tv_usec = 0;
        
        int result = select(sockfd + 1, &readfds, NULL, NULL, &timeout);
        if (result < 0) {
            perror("Select failed");
            free(packet);
            return -1;
        } else if (result == 0) {
            // Timeout, retry
            printf("Timeout, retrying... (%d/%d)\n", retries + 1, MAX_RETRIES);
            retries++;
            continue;
        }
        
        // Receive ACK
        udp_packet_t ack_packet;
        struct sockaddr_in ack_addr;
        socklen_t ack_len = sizeof(ack_addr);
        
        if (recvfrom(sockfd, &ack_packet, sizeof(ack_packet), 0,
                    (struct sockaddr*)&ack_addr, &ack_len) > 0) {
            if ((ack_packet.flags & FLAG_ACK) && 
                ntohl(ack_packet.ack_num) == seq_num) {
                // ACK received for our packet
                free(packet);
                return 0;
            }
        }
        
        retries++;
    }
    
    // Max retries exceeded
    free(packet);
    return -1;
}

int recv_reliable_udp(int sockfd, void* buffer, size_t buflen, 
                     struct sockaddr* src_addr, socklen_t* addrlen) {
    udp_packet_t* packet = malloc(sizeof(udp_packet_t) + buflen);
    
    ssize_t received = recvfrom(sockfd, packet, sizeof(udp_packet_t) + buflen, 0,
                               src_addr, addrlen);
    if (received < 0) {
        free(packet);
        return -1;
    }
    
    if (received < sizeof(udp_packet_t)) {
        // Packet too small
        free(packet);
        return -1;
    }
    
    uint16_t data_len = ntohs(packet->data_len);
    if (data_len > buflen) {
        // Data too large for buffer
        free(packet);
        return -1;
    }
    
    // Send ACK
    udp_packet_t ack_packet;
    ack_packet.seq_num = 0;
    ack_packet.ack_num = packet->seq_num;
    ack_packet.flags = htons(FLAG_ACK);
    ack_packet.data_len = 0;
    
    sendto(sockfd, &ack_packet, sizeof(udp_packet_t), 0, src_addr, *addrlen);
    
    // Copy data to buffer
    memcpy(buffer, packet->data, data_len);
    free(packet);
    
    return data_len;
}
```

### UDP Broadcast Example
```c
int setup_udp_broadcast_sender() {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        return -1;
    }
    
    // Enable broadcast
    int broadcast_enable = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_BROADCAST, 
                  &broadcast_enable, sizeof(broadcast_enable)) < 0) {
        perror("Broadcast enable failed");
        close(sockfd);
        return -1;
    }
    
    return sockfd;
}

void send_broadcast_message(int sockfd, int port, const char* message) {
    struct sockaddr_in broadcast_addr;
    
    memset(&broadcast_addr, 0, sizeof(broadcast_addr));
    broadcast_addr.sin_family = AF_INET;
    broadcast_addr.sin_port = htons(port);
    broadcast_addr.sin_addr.s_addr = INADDR_BROADCAST;
    
    if (sendto(sockfd, message, strlen(message), 0,
              (struct sockaddr*)&broadcast_addr, sizeof(broadcast_addr)) < 0) {
        perror("Broadcast send failed");
    } else {
        printf("Broadcast message sent: %s\n", message);
    }
}
```

### UDP Multicast Example
```c
#include <netinet/ip.h>

int setup_udp_multicast_sender() {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        return -1;
    }
    
    // Set multicast TTL
    unsigned char ttl = 1;
    if (setsockopt(sockfd, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl)) < 0) {
        perror("Setting multicast TTL failed");
        close(sockfd);
        return -1;
    }
    
    return sockfd;
}

int setup_udp_multicast_receiver(const char* multicast_group, int port) {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        return -1;
    }
    
    // Allow multiple listeners
    int reuse = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        perror("Setting SO_REUSEADDR failed");
        close(sockfd);
        return -1;
    }
    
    // Bind to multicast port
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    
    if (bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("Bind failed");
        close(sockfd);
        return -1;
    }
    
    // Join multicast group
    struct ip_mreq mreq;
    mreq.imr_multiaddr.s_addr = inet_addr(multicast_group);
    mreq.imr_interface.s_addr = INADDR_ANY;
    
    if (setsockopt(sockfd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        perror("Joining multicast group failed");
        close(sockfd);
        return -1;
    }
    
    return sockfd;
}
```

## UDP vs TCP Comparison

| Aspect | UDP | TCP |
|--------|-----|-----|
| Connection | Connectionless | Connection-oriented |
| Reliability | No guarantees | Reliable delivery |
| Ordering | No ordering | Ordered delivery |
| Speed | Faster | Slower due to overhead |
| Header Size | 8 bytes | 20+ bytes |
| Flow Control | None | Built-in |
| Use Cases | Gaming, streaming, DNS | Web, email, file transfer |

## Common UDP Patterns

### Request-Response Pattern
```c
typedef struct {
    uint32_t request_id;
    uint32_t timestamp;
    char data[];
} request_packet_t;

typedef struct {
    uint32_t request_id;
    uint32_t status;
    char data[];
} response_packet_t;
```

### Heartbeat/Keep-Alive Pattern
```c
void send_heartbeat(int sockfd, const struct sockaddr* dest_addr, socklen_t addrlen) {
    const char* heartbeat_msg = "HEARTBEAT";
    sendto(sockfd, heartbeat_msg, strlen(heartbeat_msg), 0, dest_addr, addrlen);
}
```

## Error Handling Best Practices

1. **Always check return values** from UDP functions
2. **Handle EAGAIN/EWOULDBLOCK** for non-blocking sockets
3. **Validate packet size** before processing
4. **Implement packet validation** to prevent corruption
5. **Handle address resolution errors** properly

## Assessment Checklist

- [ ] Can create and configure UDP sockets
- [ ] Successfully sends and receives datagrams
- [ ] Understands connectionless communication patterns
- [ ] Implements packet loss detection and handling
- [ ] Can add reliability mechanisms when needed
- [ ] Handles broadcast and multicast scenarios

## Next Steps

After mastering UDP programming:
- Explore advanced UDP topics (QUIC protocol, UDP hole punching)
- Study real-time communication protocols (RTP, RTCP)
- Learn about UDP optimization techniques

## Resources

- "UNIX Network Programming, Volume 1" by W. Richard Stevens (Chapter 8)
- [Beej's Guide - Datagram Sockets](https://beej.us/guide/bgnet/html/#datagram-sockets)
- RFC 768: User Datagram Protocol
- Linux man pages: sendto(2), recvfrom(2)

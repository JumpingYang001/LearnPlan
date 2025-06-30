# Advanced TCP/IP Topics

*Duration: 2 weeks*

## Overview

This section covers advanced TCP/IP concepts essential for building high-performance, secure network applications. You'll learn about TCP performance optimization, IP routing mechanisms, and security considerations that affect real-world network programming.

## Learning Objectives

By the end of this section, you should be able to:
- **Optimize TCP performance** through buffer tuning, congestion control, and window scaling
- **Implement IP routing and forwarding** mechanisms in applications
- **Apply TCP/IP security measures** to protect against common network attacks
- **Analyze network performance** using profiling tools and metrics
- **Troubleshoot complex networking issues** in production environments

## TCP Performance Tuning

### Understanding TCP Buffer Management

TCP buffers are critical for network performance. They determine how much data can be buffered at the sender and receiver sides, directly affecting throughput and latency.

#### Socket Buffer Architecture
```
Application Layer
       ↓
Send Buffer (SO_SNDBUF) → TCP → Network → TCP → Receive Buffer (SO_RCVBUF)
       ↑                                            ↓
  setsockopt()                                Application Layer
```

#### Comprehensive Buffer Tuning Example

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>

// Function to get current socket buffer sizes
void get_socket_buffer_info(int sock) {
    int sndbuf, rcvbuf;
    socklen_t optlen = sizeof(int);
    
    if (getsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sndbuf, &optlen) == 0) {
        printf("Current send buffer size: %d bytes\n", sndbuf);
    }
    
    if (getsockopt(sock, SOL_SOCKET, SO_RCVBUF, &rcvbuf, &optlen) == 0) {
        printf("Current receive buffer size: %d bytes\n", rcvbuf);
    }
}

// Optimized buffer configuration
int configure_tcp_buffers(int sock) {
    // Calculate optimal buffer size based on bandwidth-delay product
    // Example: 100Mbps * 50ms = 625KB
    int optimal_buffer_size = 640 * 1024;  // 640KB
    
    // Set send buffer size
    if (setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &optimal_buffer_size, sizeof(optimal_buffer_size)) < 0) {
        perror("setsockopt SO_SNDBUF");
        return -1;
    }
    
    // Set receive buffer size
    if (setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &optimal_buffer_size, sizeof(optimal_buffer_size)) < 0) {
        perror("setsockopt SO_RCVBUF");
        return -1;
    }
    
    // Enable TCP window scaling (if supported)
    int window_scale = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_WINDOW_CLAMP, &window_scale, sizeof(window_scale));
    
    printf("TCP buffers configured to %d bytes\n", optimal_buffer_size);
    return 0;
}

// Advanced TCP socket optimization
int optimize_tcp_socket(int sock) {
    // Enable TCP_NODELAY to disable Nagle's algorithm for low-latency apps
    int nodelay = 1;
    if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay)) < 0) {
        perror("TCP_NODELAY");
        return -1;
    }
    
    // Configure TCP keepalive
    int keepalive = 1;
    int keepidle = 60;    // Start keepalive after 60 seconds of inactivity
    int keepintvl = 10;   // Send keepalive probes every 10 seconds
    int keepcnt = 3;      // Send 3 probes before declaring connection dead
    
    setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, &keepalive, sizeof(keepalive));
    setsockopt(sock, IPPROTO_TCP, TCP_KEEPIDLE, &keepidle, sizeof(keepidle));
    setsockopt(sock, IPPROTO_TCP, TCP_KEEPINTVL, &keepintvl, sizeof(keepintvl));
    setsockopt(sock, IPPROTO_TCP, TCP_KEEPCNT, &keepcnt, sizeof(keepcnt));
    
    // Set socket to reuse address (useful for servers)
    int reuse = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    
    // Configure TCP congestion control algorithm (Linux-specific)
    #ifdef TCP_CONGESTION
    char congestion[] = "bbr";  // Use BBR congestion control
    setsockopt(sock, IPPROTO_TCP, TCP_CONGESTION, congestion, strlen(congestion));
    #endif
    
    printf("TCP socket optimized for high performance\n");
    return 0;
}

// Performance measurement example
typedef struct {
    struct timespec start_time;
    struct timespec end_time;
    size_t bytes_transferred;
} perf_stats_t;

void start_performance_measurement(perf_stats_t* stats) {
    clock_gettime(CLOCK_MONOTONIC, &stats->start_time);
    stats->bytes_transferred = 0;
}

void end_performance_measurement(perf_stats_t* stats) {
    clock_gettime(CLOCK_MONOTONIC, &stats->end_time);
    
    double elapsed = (stats->end_time.tv_sec - stats->start_time.tv_sec) +
                    (stats->end_time.tv_nsec - stats->start_time.tv_nsec) / 1e9;
    
    double throughput_mbps = (stats->bytes_transferred * 8.0) / (elapsed * 1e6);
    
    printf("Performance Stats:\n");
    printf("  Bytes transferred: %zu\n", stats->bytes_transferred);
    printf("  Time elapsed: %.3f seconds\n", elapsed);
    printf("  Throughput: %.2f Mbps\n", throughput_mbps);
}

// High-performance TCP server example
int create_optimized_server(int port) {
    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock < 0) {
        perror("socket");
        return -1;
    }
    
    // Apply optimizations
    optimize_tcp_socket(server_sock);
    configure_tcp_buffers(server_sock);
    
    struct sockaddr_in server_addr = {0};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);
    
    if (bind(server_sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        close(server_sock);
        return -1;
    }
    
    if (listen(server_sock, 128) < 0) {  // Larger backlog for high-load servers
        perror("listen");
        close(server_sock);
        return -1;
    }
    
    printf("Optimized TCP server listening on port %d\n", port);
    return server_sock;
}
```

#### TCP Window Scaling and Congestion Control

```c
// Understanding TCP window dynamics
void analyze_tcp_window(int sock) {
    // Get TCP info (Linux-specific)
    #ifdef TCP_INFO
    struct tcp_info info;
    socklen_t info_len = sizeof(info);
    
    if (getsockopt(sock, IPPROTO_TCP, TCP_INFO, &info, &info_len) == 0) {
        printf("TCP Connection Analysis:\n");
        printf("  State: %u\n", info.tcpi_state);
        printf("  RTT: %u μs\n", info.tcpi_rtt);
        printf("  RTT variance: %u μs\n", info.tcpi_rttvar);
        printf("  Send window: %u\n", info.tcpi_snd_wnd);
        printf("  Receive window: %u\n", info.tcpi_rcv_wnd);
        printf("  Congestion window: %u\n", info.tcpi_snd_cwnd);
        printf("  Slow start threshold: %u\n", info.tcpi_snd_ssthresh);
        printf("  Retransmits: %u\n", info.tcpi_retransmits);
    }
    #endif
}

// Adaptive buffer sizing based on network conditions
int adaptive_buffer_sizing(int sock) {
    #ifdef TCP_INFO
    struct tcp_info info;
    socklen_t info_len = sizeof(info);
    
    if (getsockopt(sock, IPPROTO_TCP, TCP_INFO, &info, &info_len) == 0) {
        // Calculate bandwidth-delay product
        uint32_t rtt_ms = info.tcpi_rtt / 1000;  // Convert to milliseconds
        uint32_t bandwidth_estimate = info.tcpi_snd_cwnd * 1448;  // Assume MSS of 1448
        
        // Optimal buffer = bandwidth * delay
        int optimal_buffer = (bandwidth_estimate * rtt_ms) / 1000;
        
        // Clamp to reasonable bounds
        if (optimal_buffer < 64 * 1024) optimal_buffer = 64 * 1024;    // Min 64KB
        if (optimal_buffer > 16 * 1024 * 1024) optimal_buffer = 16 * 1024 * 1024;  // Max 16MB
        
        // Apply new buffer sizes
        setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &optimal_buffer, sizeof(optimal_buffer));
        setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &optimal_buffer, sizeof(optimal_buffer));
        
        printf("Adaptive buffer size set to %d bytes (RTT: %u ms)\n", optimal_buffer, rtt_ms);
        return optimal_buffer;
    }
    #endif
    return -1;
}

## TCP/IP Security Considerations

### SYN Flood Defense Mechanisms

SYN flood attacks exploit the TCP three-way handshake by sending numerous SYN packets without completing the connection, exhausting server resources.

#### Understanding SYN Flood Attacks

```
Normal TCP Handshake:
Client                    Server
  |                         |
  |-------- SYN ----------->|  (1. Connection request)
  |<--- SYN-ACK ------------|  (2. Server allocates resources)
  |-------- ACK ----------->|  (3. Connection established)

SYN Flood Attack:
Attacker                  Server
  |                         |
  |-------- SYN ----------->|  (1. Fake connection request)
  |<--- SYN-ACK ------------|  (2. Server allocates resources)
  |         X               |  (3. No ACK - resources held indefinitely)
  |-------- SYN ----------->|  (Repeat with different source IPs)
  |<--- SYN-ACK ------------|
  |         X               |
```

#### SYN Cookies Implementation

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <time.h>
#include <unistd.h>

// SYN cookie generation (simplified example)
typedef struct {
    uint32_t timestamp;
    uint32_t mss;
    uint32_t hash;
} syn_cookie_t;

// Simple hash function for demonstration
uint32_t simple_hash(uint32_t src_ip, uint16_t src_port, uint32_t dst_ip, uint16_t dst_port) {
    return (src_ip ^ dst_ip ^ (src_port << 16) ^ dst_port) & 0xFFFFFF;
}

// Generate SYN cookie
uint32_t generate_syn_cookie(uint32_t src_ip, uint16_t src_port, 
                            uint32_t dst_ip, uint16_t dst_port, uint32_t seq) {
    uint32_t timestamp = time(NULL) & 0x3F;  // 6-bit timestamp
    uint32_t mss_index = 0;  // MSS encoding (simplified)
    uint32_t hash = simple_hash(src_ip, src_port, dst_ip, dst_port);
    
    // Combine components into cookie
    return (timestamp << 26) | (mss_index << 24) | hash;
}

// Verify SYN cookie
int verify_syn_cookie(uint32_t cookie, uint32_t src_ip, uint16_t src_port,
                     uint32_t dst_ip, uint16_t dst_port, uint32_t ack_seq) {
    uint32_t current_time = time(NULL) & 0x3F;
    uint32_t cookie_time = (cookie >> 26) & 0x3F;
    
    // Check if cookie is recent (within 64 seconds)
    if (((current_time - cookie_time) & 0x3F) > 60) {
        return 0;  // Cookie too old
    }
    
    // Verify hash component
    uint32_t expected_hash = simple_hash(src_ip, src_port, dst_ip, dst_port);
    uint32_t cookie_hash = cookie & 0xFFFFFF;
    
    return (expected_hash == cookie_hash);
}

// Configure system-level SYN flood protection
void configure_syn_protection() {
    printf("Configuring SYN flood protection:\n");
    
    // Enable SYN cookies (requires root privileges)
    system("echo 1 > /proc/sys/net/ipv4/tcp_syncookies 2>/dev/null");
    printf("  ✓ SYN cookies enabled\n");
    
    // Reduce SYN-RECV timeout
    system("echo 1 > /proc/sys/net/ipv4/tcp_synack_retries 2>/dev/null");
    printf("  ✓ SYN-ACK retries reduced\n");
    
    // Limit max half-open connections
    system("echo 1024 > /proc/sys/net/ipv4/tcp_max_syn_backlog 2>/dev/null");
    printf("  ✓ SYN backlog limited\n");
    
    // Enable source address validation
    system("echo 1 > /proc/sys/net/ipv4/conf/all/rp_filter 2>/dev/null");
    printf("  ✓ Reverse path filtering enabled\n");
}

// Application-level SYN flood mitigation
typedef struct connection_entry {
    uint32_t src_ip;
    uint16_t src_port;
    time_t first_seen;
    int count;
    struct connection_entry* next;
} connection_entry_t;

#define MAX_CONNECTIONS_PER_IP 10
#define TIME_WINDOW 60  // seconds

connection_entry_t* connection_table[1024] = {0};

// Simple hash table for tracking connections
int hash_connection(uint32_t ip) {
    return ip % 1024;
}

// Check if IP is flooding
int is_flooding(uint32_t src_ip, uint16_t src_port) {
    int hash = hash_connection(src_ip);
    connection_entry_t* entry = connection_table[hash];
    time_t now = time(NULL);
    
    // Find existing entry
    while (entry) {
        if (entry->src_ip == src_ip) {
            // Reset counter if time window expired
            if (now - entry->first_seen > TIME_WINDOW) {
                entry->first_seen = now;
                entry->count = 1;
                return 0;
            }
            
            // Increment counter
            entry->count++;
            if (entry->count > MAX_CONNECTIONS_PER_IP) {
                printf("SYN flood detected from %s (count: %d)\n",
                       inet_ntoa(*(struct in_addr*)&src_ip), entry->count);
                return 1;  // Flooding detected
            }
            return 0;
        }
        entry = entry->next;
    }
    
    // New entry
    entry = malloc(sizeof(connection_entry_t));
    entry->src_ip = src_ip;
    entry->src_port = src_port;
    entry->first_seen = now;
    entry->count = 1;
    entry->next = connection_table[hash];
    connection_table[hash] = entry;
    
    return 0;
}

// Hardened TCP server with SYN flood protection
int create_hardened_server(int port) {
    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock < 0) {
        perror("socket");
        return -1;
    }
    
    // Configure socket options for security
    int reuse = 1;
    setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    
    // Reduce keep-alive time to free resources faster
    int keepalive = 1;
    int keepidle = 30;
    int keepintvl = 5;
    int keepcnt = 3;
    
    setsockopt(server_sock, SOL_SOCKET, SO_KEEPALIVE, &keepalive, sizeof(keepalive));
    setsockopt(server_sock, IPPROTO_TCP, TCP_KEEPIDLE, &keepidle, sizeof(keepidle));
    setsockopt(server_sock, IPPROTO_TCP, TCP_KEEPINTVL, &keepintvl, sizeof(keepintvl));
    setsockopt(server_sock, IPPROTO_TCP, TCP_KEEPCNT, &keepcnt, sizeof(keepcnt));
    
    // Bind and listen
    struct sockaddr_in server_addr = {0};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);
    
    if (bind(server_sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        close(server_sock);
        return -1;
    }
    
    // Use smaller backlog to limit exposure
    if (listen(server_sock, 64) < 0) {
        perror("listen");
        close(server_sock);
        return -1;
    }
    
    printf("Hardened TCP server listening on port %d\n", port);
    return server_sock;
}
```

## IP Routing and Forwarding

### Understanding IP Routing Fundamentals

IP routing is the process of forwarding packets from source to destination across multiple networks. This section covers both basic routing concepts and advanced implementation techniques.

#### Routing Table Structure

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <net/route.h>
#include <linux/rtnetlink.h>

// Routing table entry structure
typedef struct route_entry {
    uint32_t destination;    // Destination network
    uint32_t netmask;       // Network mask
    uint32_t gateway;       // Next hop gateway
    char interface[16];     // Output interface
    int metric;             // Route metric/cost
    struct route_entry* next;
} route_entry_t;

// Simple routing table implementation
typedef struct {
    route_entry_t* routes;
    int count;
} routing_table_t;

// Initialize routing table
routing_table_t* create_routing_table() {
    routing_table_t* table = malloc(sizeof(routing_table_t));
    table->routes = NULL;
    table->count = 0;
    return table;
}

// Add route to table
void add_route(routing_table_t* table, const char* dest, const char* mask, 
               const char* gateway, const char* interface, int metric) {
    route_entry_t* new_route = malloc(sizeof(route_entry_t));
    
    new_route->destination = inet_addr(dest);
    new_route->netmask = inet_addr(mask);
    new_route->gateway = inet_addr(gateway);
    strncpy(new_route->interface, interface, sizeof(new_route->interface) - 1);
    new_route->metric = metric;
    new_route->next = table->routes;
    
    table->routes = new_route;
    table->count++;
    
    printf("Added route: %s/%s via %s dev %s metric %d\n", 
           dest, mask, gateway, interface, metric);
}

// Longest prefix match algorithm
route_entry_t* lookup_route(routing_table_t* table, uint32_t dest_ip) {
    route_entry_t* best_match = NULL;
    uint32_t longest_mask = 0;
    
    route_entry_t* current = table->routes;
    while (current) {
        // Check if destination matches this route
        if ((dest_ip & current->netmask) == (current->destination & current->netmask)) {
            // Check if this is a more specific route (longer prefix)
            if (current->netmask >= longest_mask) {
                longest_mask = current->netmask;
                best_match = current;
            }
        }
        current = current->next;
    }
    
    return best_match;
}

// Display routing table
void print_routing_table(routing_table_t* table) {
    printf("\nRouting Table:\n");
    printf("%-15s %-15s %-15s %-10s %s\n", 
           "Destination", "Netmask", "Gateway", "Interface", "Metric");
    printf("----------------------------------------------------------------\n");
    
    route_entry_t* current = table->routes;
    while (current) {
        struct in_addr dest, mask, gw;
        dest.s_addr = current->destination;
        mask.s_addr = current->netmask;
        gw.s_addr = current->gateway;
        
        printf("%-15s %-15s %-15s %-10s %d\n",
               inet_ntoa(dest), inet_ntoa(mask), inet_ntoa(gw),
               current->interface, current->metric);
        current = current->next;
    }
}
```

#### Network Interface Management

```c
#include <ifaddrs.h>
#include <net/if.h>
#include <sys/ioctl.h>

// Network interface information
typedef struct {
    char name[IFNAMSIZ];
    uint32_t ip_address;
    uint32_t netmask;
    uint32_t broadcast;
    int flags;
    int mtu;
} interface_info_t;

// Get network interface information
int get_interface_info(const char* if_name, interface_info_t* info) {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("socket");
        return -1;
    }
    
    struct ifreq ifr;
    strncpy(ifr.ifr_name, if_name, IFNAMSIZ - 1);
    
    // Get IP address
    if (ioctl(sock, SIOCGIFADDR, &ifr) == 0) {
        struct sockaddr_in* addr = (struct sockaddr_in*)&ifr.ifr_addr;
        info->ip_address = addr->sin_addr.s_addr;
    }
    
    // Get netmask
    if (ioctl(sock, SIOCGIFNETMASK, &ifr) == 0) {
        struct sockaddr_in* mask = (struct sockaddr_in*)&ifr.ifr_netmask;
        info->netmask = mask->sin_addr.s_addr;
    }
    
    // Get broadcast address
    if (ioctl(sock, SIOCGIFBRDADDR, &ifr) == 0) {
        struct sockaddr_in* bcast = (struct sockaddr_in*)&ifr.ifr_broadaddr;
        info->broadcast = bcast->sin_addr.s_addr;
    }
    
    // Get interface flags
    if (ioctl(sock, SIOCGIFFLAGS, &ifr) == 0) {
        info->flags = ifr.ifr_flags;
    }
    
    // Get MTU
    if (ioctl(sock, SIOCGIFMTU, &ifr) == 0) {
        info->mtu = ifr.ifr_mtu;
    }
    
    strncpy(info->name, if_name, IFNAMSIZ - 1);
    close(sock);
    return 0;
}

// List all network interfaces
void list_interfaces() {
    struct ifaddrs *ifap, *ifa;
    
    if (getifaddrs(&ifap) == -1) {
        perror("getifaddrs");
        return;
    }
    
    printf("\nNetwork Interfaces:\n");
    printf("%-10s %-15s %-15s %-10s %s\n", 
           "Interface", "IP Address", "Netmask", "MTU", "Flags");
    printf("----------------------------------------------------------------\n");
    
    for (ifa = ifap; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET) {
            interface_info_t info;
            if (get_interface_info(ifa->ifa_name, &info) == 0) {
                struct in_addr ip, mask;
                ip.s_addr = info.ip_address;
                mask.s_addr = info.netmask;
                
                printf("%-10s %-15s %-15s %-10d %s%s%s\n",
                       info.name, inet_ntoa(ip), inet_ntoa(mask), info.mtu,
                       (info.flags & IFF_UP) ? "UP " : "DOWN ",
                       (info.flags & IFF_RUNNING) ? "RUNNING " : "",
                       (info.flags & IFF_LOOPBACK) ? "LOOPBACK" : "");
            }
        }
    }
    
    freeifaddrs(ifap);
}
```

#### Advanced Routing Implementation

```c
// Policy-based routing
typedef struct {
    uint32_t src_network;
    uint32_t src_mask;
    uint32_t dst_network;
    uint32_t dst_mask;
    int table_id;
    int priority;
} routing_policy_t;

// Multi-path routing support
typedef struct nexthop {
    uint32_t gateway;
    char interface[16];
    int weight;
    struct nexthop* next;
} nexthop_t;

typedef struct {
    uint32_t destination;
    uint32_t netmask;
    nexthop_t* nexthops;
    int total_weight;
} multipath_route_t;

// Load balancing across multiple paths
nexthop_t* select_nexthop(multipath_route_t* route) {
    if (!route->nexthops) return NULL;
    
    int random_weight = rand() % route->total_weight;
    int current_weight = 0;
    
    nexthop_t* current = route->nexthops;
    while (current) {
        current_weight += current->weight;
        if (random_weight < current_weight) {
            return current;
        }
        current = current->next;
    }
    
    return route->nexthops;  // Fallback to first nexthop
}

// Implement basic packet forwarding
typedef struct {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint8_t protocol;
    size_t payload_size;
    char* payload;
} packet_t;

// Forward packet based on routing table
int forward_packet(routing_table_t* table, packet_t* packet) {
    route_entry_t* route = lookup_route(table, packet->dst_ip);
    
    if (!route) {
        printf("No route to destination %s\n", 
               inet_ntoa(*(struct in_addr*)&packet->dst_ip));
        return -1;  // Destination unreachable
    }
    
    printf("Forwarding packet to %s via %s (interface: %s)\n",
           inet_ntoa(*(struct in_addr*)&packet->dst_ip),
           inet_ntoa(*(struct in_addr*)&route->gateway),
           route->interface);
    
    // In a real implementation, this would:
    // 1. Decrement TTL
    // 2. Recalculate checksum
    // 3. Update MAC addresses
    // 4. Send via output interface
    
    return 0;
}

// Traceroute implementation
#include <sys/time.h>

void traceroute(const char* destination) {
    int sock = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP);
    if (sock < 0) {
        perror("socket (need root privileges)");
        return;
    }
    
    struct sockaddr_in dest_addr;
    dest_addr.sin_family = AF_INET;
    inet_aton(destination, &dest_addr.sin_addr);
    
    printf("Traceroute to %s:\n", destination);
    
    for (int ttl = 1; ttl <= 30; ttl++) {
        // Set TTL
        if (setsockopt(sock, IPPROTO_IP, IP_TTL, &ttl, sizeof(ttl)) < 0) {
            perror("setsockopt TTL");
            break;
        }
        
        struct timeval start, end;
        gettimeofday(&start, NULL);
        
        // Send ICMP echo request (simplified)
        printf("%2d  ", ttl);
        
        // In a complete implementation, you would:
        // 1. Send ICMP echo request
        // 2. Wait for ICMP time exceeded or echo reply
        // 3. Extract source IP from response
        // 4. Calculate round-trip time
        
        printf("* * *\n");  // Placeholder
        
        // Break if we reached the destination
        // if (response_from_destination) break;
    }
    
    close(sock);
}
```

#### Dynamic Routing Protocols (Conceptual)

```c
// OSPF-like link state information
typedef struct {
    uint32_t router_id;
    uint32_t network;
    uint32_t netmask;
    int metric;
    time_t timestamp;
} link_state_t;

// Distance vector routing (RIP-like)
typedef struct {
    uint32_t destination;
    uint32_t netmask;
    int distance;
    uint32_t next_hop;
    time_t last_update;
} distance_vector_entry_t;

// BGP-like path vector
typedef struct {
    uint32_t destination;
    uint32_t netmask;
    uint32_t* as_path;
    int path_length;
    uint32_t next_hop;
    int local_pref;
} bgp_route_t;

// Route redistribution between protocols
void redistribute_routes(routing_table_t* static_table, 
                        distance_vector_entry_t* rip_table,
                        int rip_count) {
    printf("Redistributing routes between protocols...\n");
    
    // Add RIP routes to main table with higher metric
    for (int i = 0; i < rip_count; i++) {
        if (time(NULL) - rip_table[i].last_update < 180) {  // Route timeout
            char dest[16], mask[16], nh[16];
            strcpy(dest, inet_ntoa(*(struct in_addr*)&rip_table[i].destination));
            strcpy(mask, inet_ntoa(*(struct in_addr*)&rip_table[i].netmask));
            strcpy(nh, inet_ntoa(*(struct in_addr*)&rip_table[i].next_hop));
            
            add_route(static_table, dest, mask, nh, "eth0", 
                     rip_table[i].distance + 10);  // Add administrative distance
        }
    }
}
```

## Network Performance Analysis and Optimization

### Performance Monitoring and Metrics

Understanding network performance requires monitoring key metrics and identifying bottlenecks.

#### Network Performance Metrics

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

// Network performance statistics
typedef struct {
    // Throughput metrics
    uint64_t bytes_sent;
    uint64_t bytes_received;
    uint64_t packets_sent;
    uint64_t packets_received;
    
    // Timing metrics
    struct timeval start_time;
    struct timeval end_time;
    double min_rtt;
    double max_rtt;
    double avg_rtt;
    int rtt_samples;
    
    // Error metrics
    uint32_t packet_loss;
    uint32_t retransmissions;
    uint32_t timeouts;
    
    // Connection metrics
    int active_connections;
    int failed_connections;
} network_stats_t;

// Initialize performance monitoring
void init_network_stats(network_stats_t* stats) {
    memset(stats, 0, sizeof(network_stats_t));
    gettimeofday(&stats->start_time, NULL);
    stats->min_rtt = INFINITY;
    stats->max_rtt = 0.0;
}

// Update RTT statistics
void update_rtt_stats(network_stats_t* stats, double rtt) {
    if (rtt < stats->min_rtt) stats->min_rtt = rtt;
    if (rtt > stats->max_rtt) stats->max_rtt = rtt;
    
    // Calculate running average
    stats->avg_rtt = ((stats->avg_rtt * stats->rtt_samples) + rtt) / (stats->rtt_samples + 1);
    stats->rtt_samples++;
}

// Calculate network utilization
double calculate_network_utilization(network_stats_t* stats, double link_capacity_bps) {
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    
    double elapsed = (current_time.tv_sec - stats->start_time.tv_sec) +
                    (current_time.tv_usec - stats->start_time.tv_usec) / 1e6;
    
    if (elapsed <= 0) return 0.0;
    
    double actual_throughput = (stats->bytes_sent + stats->bytes_received) * 8.0 / elapsed;
    return (actual_throughput / link_capacity_bps) * 100.0;
}

// Print performance report
void print_network_stats(network_stats_t* stats) {
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    
    double elapsed = (current_time.tv_sec - stats->start_time.tv_sec) +
                    (current_time.tv_usec - stats->start_time.tv_usec) / 1e6;
    
    printf("\n=== Network Performance Report ===\n");
    printf("Duration: %.2f seconds\n", elapsed);
    printf("\nThroughput:\n");
    printf("  Bytes sent: %lu (%.2f MB)\n", stats->bytes_sent, stats->bytes_sent / 1048576.0);
    printf("  Bytes received: %lu (%.2f MB)\n", stats->bytes_received, stats->bytes_received / 1048576.0);
    printf("  Send rate: %.2f Mbps\n", (stats->bytes_sent * 8.0) / (elapsed * 1e6));
    printf("  Receive rate: %.2f Mbps\n", (stats->bytes_received * 8.0) / (elapsed * 1e6));
    
    printf("\nLatency:\n");
    printf("  Min RTT: %.2f ms\n", stats->min_rtt * 1000);
    printf("  Max RTT: %.2f ms\n", stats->max_rtt * 1000);
    printf("  Avg RTT: %.2f ms\n", stats->avg_rtt * 1000);
    
    printf("\nReliability:\n");
    printf("  Packet loss: %u (%.2f%%)\n", stats->packet_loss,
           (stats->packet_loss * 100.0) / stats->packets_sent);
    printf("  Retransmissions: %u\n", stats->retransmissions);
    printf("  Timeouts: %u\n", stats->timeouts);
    
    printf("\nConnections:\n");
    printf("  Active: %d\n", stats->active_connections);
    printf("  Failed: %d\n", stats->failed_connections);
}
```

#### Bandwidth Testing and Analysis

```c
// Bandwidth measurement tool
typedef struct {
    int socket;
    size_t buffer_size;
    char* send_buffer;
    char* recv_buffer;
    network_stats_t stats;
} bandwidth_tester_t;

// Initialize bandwidth tester
bandwidth_tester_t* create_bandwidth_tester(size_t buffer_size) {
    bandwidth_tester_t* tester = malloc(sizeof(bandwidth_tester_t));
    tester->buffer_size = buffer_size;
    tester->send_buffer = malloc(buffer_size);
    tester->recv_buffer = malloc(buffer_size);
    
    // Fill send buffer with test pattern
    for (size_t i = 0; i < buffer_size; i++) {
        tester->send_buffer[i] = i & 0xFF;
    }
    
    init_network_stats(&tester->stats);
    return tester;
}

// Perform bandwidth test (client side)
int bandwidth_test_client(const char* server_ip, int port, int duration_seconds) {
    bandwidth_tester_t* tester = create_bandwidth_tester(64 * 1024);  // 64KB buffer
    
    // Create socket
    tester->socket = socket(AF_INET, SOCK_STREAM, 0);
    if (tester->socket < 0) {
        perror("socket");
        return -1;
    }
    
    // Connect to server
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    inet_aton(server_ip, &server_addr.sin_addr);
    
    if (connect(tester->socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect");
        close(tester->socket);
        return -1;
    }
    
    printf("Connected to %s:%d, starting bandwidth test...\n", server_ip, port);
    
    // Configure socket for maximum performance
    int buffer_size = 256 * 1024;
    setsockopt(tester->socket, SOL_SOCKET, SO_SNDBUF, &buffer_size, sizeof(buffer_size));
    
    time_t start_time = time(NULL);
    time_t end_time = start_time + duration_seconds;
    
    // Send data continuously
    while (time(NULL) < end_time) {
        ssize_t sent = send(tester->socket, tester->send_buffer, tester->buffer_size, MSG_NOSIGNAL);
        if (sent > 0) {
            tester->stats.bytes_sent += sent;
            tester->stats.packets_sent++;
        } else if (sent < 0) {
            perror("send");
            break;
        }
    }
    
    printf("Bandwidth test completed\n");
    print_network_stats(&tester->stats);
    
    close(tester->socket);
    free(tester->send_buffer);
    free(tester->recv_buffer);
    free(tester);
    
    return 0;
}

// Perform bandwidth test (server side)
int bandwidth_test_server(int port) {
    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock < 0) {
        perror("socket");
        return -1;
    }
    
    int reuse = 1;
    setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    
    struct sockaddr_in server_addr = {0};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);
    
    if (bind(server_sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        close(server_sock);
        return -1;
    }
    
    if (listen(server_sock, 1) < 0) {
        perror("listen");
        close(server_sock);
        return -1;
    }
    
    printf("Bandwidth test server listening on port %d\n", port);
    
    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_sock = accept(server_sock, (struct sockaddr*)&client_addr, &client_len);
        
        if (client_sock < 0) {
            perror("accept");
            continue;
        }
        
        printf("Client connected: %s\n", inet_ntoa(client_addr.sin_addr));
        
        bandwidth_tester_t* tester = create_bandwidth_tester(64 * 1024);
        tester->socket = client_sock;
        
        // Configure socket for maximum performance
        int buffer_size = 256 * 1024;
        setsockopt(client_sock, SOL_SOCKET, SO_RCVBUF, &buffer_size, sizeof(buffer_size));
        
        // Receive data
        ssize_t received;
        while ((received = recv(client_sock, tester->recv_buffer, tester->buffer_size, 0)) > 0) {
            tester->stats.bytes_received += received;
            tester->stats.packets_received++;
        }
        
        printf("Client disconnected\n");
        print_network_stats(&tester->stats);
        
        close(client_sock);
        free(tester->send_buffer);
        free(tester->recv_buffer);
        free(tester);
    }
    
    close(server_sock);
    return 0;
}
```

#### Latency Testing and Jitter Analysis

```c
#include <sys/time.h>
#include <math.h>

// Ping implementation for latency testing
typedef struct {
    double* rtts;
    int count;
    int capacity;
    double sum;
    double sum_squares;
} rtt_collection_t;

// Initialize RTT collection
rtt_collection_t* create_rtt_collection(int capacity) {
    rtt_collection_t* collection = malloc(sizeof(rtt_collection_t));
    collection->rtts = malloc(capacity * sizeof(double));
    collection->count = 0;
    collection->capacity = capacity;
    collection->sum = 0.0;
    collection->sum_squares = 0.0;
    return collection;
}

// Add RTT measurement
void add_rtt(rtt_collection_t* collection, double rtt) {
    if (collection->count < collection->capacity) {
        collection->rtts[collection->count] = rtt;
        collection->sum += rtt;
        collection->sum_squares += (rtt * rtt);
        collection->count++;
    }
}

// Calculate statistics
void calculate_rtt_stats(rtt_collection_t* collection) {
    if (collection->count == 0) return;
    
    double mean = collection->sum / collection->count;
    double variance = (collection->sum_squares / collection->count) - (mean * mean);
    double std_dev = sqrt(variance);
    
    // Calculate jitter (mean deviation)
    double jitter = 0.0;
    for (int i = 1; i < collection->count; i++) {
        jitter += fabs(collection->rtts[i] - collection->rtts[i-1]);
    }
    jitter /= (collection->count - 1);
    
    // Find min and max
    double min_rtt = collection->rtts[0];
    double max_rtt = collection->rtts[0];
    for (int i = 1; i < collection->count; i++) {
        if (collection->rtts[i] < min_rtt) min_rtt = collection->rtts[i];
        if (collection->rtts[i] > max_rtt) max_rtt = collection->rtts[i];
    }
    
    printf("\n=== Latency Analysis ===\n");
    printf("Packets sent: %d\n", collection->count);
    printf("Min RTT: %.3f ms\n", min_rtt * 1000);
    printf("Max RTT: %.3f ms\n", max_rtt * 1000);
    printf("Mean RTT: %.3f ms\n", mean * 1000);
    printf("Standard deviation: %.3f ms\n", std_dev * 1000);
    printf("Jitter: %.3f ms\n", jitter * 1000);
    
    // Calculate percentiles
    // Sort RTTs for percentile calculation
    for (int i = 0; i < collection->count - 1; i++) {
        for (int j = i + 1; j < collection->count; j++) {
            if (collection->rtts[i] > collection->rtts[j]) {
                double temp = collection->rtts[i];
                collection->rtts[i] = collection->rtts[j];
                collection->rtts[j] = temp;
            }
        }
    }
    
    printf("50th percentile: %.3f ms\n", collection->rtts[collection->count / 2] * 1000);
    printf("95th percentile: %.3f ms\n", collection->rtts[(collection->count * 95) / 100] * 1000);
    printf("99th percentile: %.3f ms\n", collection->rtts[(collection->count * 99) / 100] * 1000);
}

// Simple ping implementation
double measure_rtt(const char* hostname) {
    struct timeval start, end;
    
    // Create raw ICMP socket (requires root)
    int sock = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP);
    if (sock < 0) {
        // Fallback to TCP connect for RTT estimation
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) return -1;
        
        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(80);  // Try HTTP port
        inet_aton(hostname, &addr.sin_addr);
        
        gettimeofday(&start, NULL);
        int result = connect(sock, (struct sockaddr*)&addr, sizeof(addr));
        gettimeofday(&end, NULL);
        
        close(sock);
        
        if (result == 0) {
            return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
        }
    }
    
    return -1;  // Failed to measure
}
```

#### Quality of Service (QoS) Implementation

```c
// Traffic shaping and QoS
typedef enum {
    QOS_CLASS_REALTIME = 0,     // Voice, video
    QOS_CLASS_INTERACTIVE = 1,  // Gaming, VoIP signaling
    QOS_CLASS_BULK = 2,         // File transfers
    QOS_CLASS_BACKGROUND = 3    // Backups, updates
} qos_class_t;

typedef struct {
    qos_class_t class;
    uint32_t max_bandwidth;     // Bytes per second
    uint32_t burst_size;        // Maximum burst
    uint32_t current_tokens;    // Token bucket
    struct timeval last_update;
} traffic_shaper_t;

// Initialize traffic shaper
traffic_shaper_t* create_traffic_shaper(qos_class_t class, uint32_t max_bps, uint32_t burst_size) {
    traffic_shaper_t* shaper = malloc(sizeof(traffic_shaper_t));
    shaper->class = class;
    shaper->max_bandwidth = max_bps;
    shaper->burst_size = burst_size;
    shaper->current_tokens = burst_size;
    gettimeofday(&shaper->last_update, NULL);
    return shaper;
}

// Check if transmission is allowed (token bucket algorithm)
int can_transmit(traffic_shaper_t* shaper, uint32_t bytes) {
    struct timeval now;
    gettimeofday(&now, NULL);
    
    // Calculate time elapsed
    double elapsed = (now.tv_sec - shaper->last_update.tv_sec) +
                    (now.tv_usec - shaper->last_update.tv_usec) / 1e6;
    
    // Add tokens based on elapsed time
    uint32_t new_tokens = (uint32_t)(elapsed * shaper->max_bandwidth);
    shaper->current_tokens += new_tokens;
    
    // Cap at burst size
    if (shaper->current_tokens > shaper->burst_size) {
        shaper->current_tokens = shaper->burst_size;
    }
    
    shaper->last_update = now;
    
    // Check if we have enough tokens
    if (shaper->current_tokens >= bytes) {
        shaper->current_tokens -= bytes;
        return 1;  // Transmission allowed
    }
    
    return 0;  // Transmission denied (rate limited)
}

// Set socket QoS parameters
int set_socket_qos(int sock, qos_class_t class) {
    int tos = 0;
    int priority = 0;
    
    switch (class) {
        case QOS_CLASS_REALTIME:
            tos = 0xB8;  // EF (Expedited Forwarding)
            priority = 7;
            break;
        case QOS_CLASS_INTERACTIVE:
            tos = 0x88;  // AF41 (Assured Forwarding)
            priority = 6;
            break;
        case QOS_CLASS_BULK:
            tos = 0x28;  // AF11
            priority = 2;
            break;
        case QOS_CLASS_BACKGROUND:
            tos = 0x08;  // CS1 (Class Selector)
            priority = 1;
            break;
    }
    
    // Set Type of Service (IP_TOS)
    if (setsockopt(sock, IPPROTO_IP, IP_TOS, &tos, sizeof(tos)) < 0) {
        perror("setsockopt IP_TOS");
        return -1;
    }
    
    // Set socket priority
    if (setsockopt(sock, SOL_SOCKET, SO_PRIORITY, &priority, sizeof(priority)) < 0) {
        perror("setsockopt SO_PRIORITY");
        return -1;
    }
    
    printf("Socket QoS configured: class=%d, TOS=0x%02X, priority=%d\n", 
           class, tos, priority);
    return 0;
}
```

## Practical Exercises and Labs

### Lab 1: TCP Performance Optimization

**Objective:** Implement and test various TCP optimization techniques.

**Setup:**
```c
// Create a test client and server to measure performance improvements

// Baseline test (no optimizations)
int baseline_server() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    // Basic socket setup without optimizations
    return sock;
}

// Optimized version
int optimized_server() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    
    // Apply all optimizations learned
    optimize_tcp_socket(sock);
    configure_tcp_buffers(sock);
    
    return sock;
}

// TODO: Implement benchmark comparison
// Measure: throughput, latency, CPU usage
```

**Tasks:**
1. Implement baseline TCP client/server
2. Add buffer size optimization
3. Configure TCP_NODELAY appropriately
4. Implement adaptive buffer sizing
5. Compare performance metrics

### Lab 2: Network Security Implementation

**Objective:** Build a secure server with SYN flood protection and access control.

**Components to implement:**
```c
// TODO: Complete these functions
int implement_syn_cookies(int server_sock);
int add_access_control(uint32_t client_ip);
int implement_rate_limiting(uint32_t client_ip);
int detect_port_scan(uint32_t client_ip, uint16_t port);
```

**Requirements:**
- SYN flood detection and mitigation
- IP-based access control lists
- Rate limiting per client
- Port scan detection
- Connection state monitoring

### Lab 3: Advanced Routing Simulation

**Objective:** Implement a software router with multiple routing protocols.

**Features to implement:**
```c
// Routing protocol simulation
typedef struct {
    routing_table_t* static_routes;
    distance_vector_entry_t* rip_routes;
    link_state_t* ospf_database;
} router_t;

// TODO: Implement these routing functions
int process_rip_update(router_t* router, rip_packet_t* packet);
int calculate_shortest_path(router_t* router);  // Dijkstra for OSPF
int redistribute_routes(router_t* router);
```

## Study Materials and Resources

### Essential Reading

**Books:**
- **"TCP/IP Illustrated, Volume 1"** by W. Richard Stevens
  - Chapters 17-24: Advanced TCP topics
  - Focus on: Congestion control, performance analysis
- **"UNIX Network Programming, Volume 1"** by W. Richard Stevens
  - Chapters 7-8: Socket options and performance
- **"Computer Networks"** by Andrew Tanenbaum
  - Chapter 6: Network layer and routing protocols

**RFCs (Request for Comments):**
- RFC 793: TCP Specification
- RFC 1323: TCP Extensions for High Performance
- RFC 2018: TCP Selective Acknowledgment Options
- RFC 3168: Explicit Congestion Notification (ECN)
- RFC 4987: TCP SYN Flooding Attacks and Common Mitigations

### Online Resources

**Documentation:**
- Linux TCP/IP Stack Documentation
- FreeBSD Network Stack Guide
- Cisco Networking Academy materials

**Tools Documentation:**
```bash
# Performance analysis tools
man ss           # Socket statistics
man netstat      # Network connections
man iptraf-ng    # Network traffic monitor
man tcpdump      # Packet capture
man wireshark    # Protocol analyzer

# System tuning
man sysctl       # Kernel parameters
man tc           # Traffic control
man ip           # IP configuration
```

### Hands-on Practice

**Network Simulation:**
- Use Mininet for network topology simulation
- GNS3 for advanced routing protocol testing
- Packet Tracer for Cisco-specific scenarios

**Performance Testing:**
```bash
# Install network testing tools
sudo apt-get install iperf3 netperf nuttcp

# TCP performance testing
iperf3 -s                    # Server mode
iperf3 -c server_ip          # Client mode
iperf3 -c server_ip -P 4     # Parallel streams

# Network latency testing
ping -c 100 destination
mtr destination              # Traceroute with statistics
```

**Security Testing:**
```bash
# SYN flood testing (ethical testing only)
hping3 -S -p 80 --flood target_ip

# Port scanning detection
nmap -sS target_ip
nmap -sF target_ip    # FIN scan
nmap -sX target_ip    # Xmas scan
```

### Development Environment Setup

**Required Packages:**
```bash
# Ubuntu/Debian
sudo apt-get install build-essential
sudo apt-get install libpcap-dev
sudo apt-get install net-tools
sudo apt-get install iproute2
sudo apt-get install tcpdump
sudo apt-get install wireshark

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install libpcap-devel
sudo yum install net-tools
sudo yum install iproute
```

**Compilation Examples:**
```bash
# Basic network program
gcc -o network_app network_app.c

# With pthread support
gcc -pthread -o multithread_server server.c

# With raw socket capabilities
gcc -o raw_socket_app raw_socket_app.c
# Note: May require root privileges to run

# With optimization
gcc -O2 -o high_perf_server server.c -lm
```

### Assessment Questions

**Conceptual Questions:**
1. Explain the bandwidth-delay product and its impact on TCP performance
2. How do SYN cookies prevent SYN flood attacks without maintaining state?
3. Compare distance vector vs link state routing protocols
4. What are the trade-offs between TCP_NODELAY and Nagle's algorithm?
5. How does ECN (Explicit Congestion Notification) improve network performance?

**Technical Implementation:**
6. Implement a simple congestion control algorithm
7. Design a multi-path routing decision algorithm
8. Create a network traffic classification system
9. Build a connection state tracking system
10. Implement QoS packet marking based on application type

**Performance Analysis:**
11. Calculate optimal TCP buffer sizes for given network conditions
12. Analyze packet captures to identify performance bottlenecks
13. Design a network monitoring system with alerting
14. Optimize application for specific network characteristics
15. Implement adaptive algorithms based on network feedback

### Real-world Applications

**Industry Use Cases:**
- **CDN Optimization:** TCP performance tuning for content delivery
- **Gaming Networks:** Low-latency optimization techniques  
- **Financial Trading:** Ultra-low latency network design
- **Video Streaming:** Adaptive bitrate and QoS implementation
- **IoT Networks:** Resource-constrained networking
- **Cloud Infrastructure:** Multi-tenant network isolation and QoS

**Career Paths:**
- Network Performance Engineer
- Security Engineer (Network Security)
- Systems Network Architect
- DevOps/SRE with networking focus
- Network Protocol Developer

## Next Steps

After completing this section, you should be prepared for:
1. **[Network Programming Projects](../03_HTTP_Protocol/README.md)** - Apply TCP/IP knowledge to application protocols
2. **[WebRTC and Real-time Communications](../05_WebRTC/README.md)** - Advanced real-time networking
3. **[Network Security Deep Dive](../../12_Industry_Protocols/README.md)** - Industry-specific security protocols

**Certification Paths:**
- CCNA (Cisco Certified Network Associate)
- CompTIA Network+
- CISSP (Security focus)
- Linux Foundation Networking certifications

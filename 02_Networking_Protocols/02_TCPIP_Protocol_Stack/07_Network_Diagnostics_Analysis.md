# Network Diagnostics and Analysis

*Duration: 2 weeks*

## Overview

Network diagnostics and analysis are critical skills for understanding network behavior, troubleshooting connectivity issues, and monitoring network performance. This comprehensive guide covers essential diagnostic tools and packet analysis techniques used in professional network administration and development.

## Learning Objectives

By the end of this section, you should be able to:
- **Master essential network diagnostic tools** (ping, traceroute, netstat, ss, etc.)
- **Perform packet capture and analysis** using tcpdump and Wireshark
- **Implement custom network diagnostic tools** in C/C++
- **Analyze network protocols** at different layers of the TCP/IP stack
- **Troubleshoot common network issues** using systematic approaches
- **Monitor network performance** and identify bottlenecks
- **Use advanced analysis techniques** for security and optimization

## Essential Network Diagnostic Tools

### 1. Ping - ICMP Echo Testing

#### Understanding Ping

Ping uses ICMP Echo Request and Echo Reply messages to test network connectivity and measure round-trip time (RTT).

**Basic Ping Usage:**
```bash
# Basic ping
ping google.com

# Ping with count limit
ping -c 4 8.8.8.8

# Ping with specific interval
ping -i 0.5 192.168.1.1

# Ping with larger packet size
ping -s 1024 google.com

# Ping IPv6
ping6 2001:4860:4860::8888
```

#### Custom Ping Implementation in C

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/ip.h>
#include <netinet/ip_icmp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <time.h>
#include <sys/time.h>

#define PACKET_SIZE 64
#define MAX_WAIT_TIME 5

struct ping_packet {
    struct icmphdr hdr;
    char msg[PACKET_SIZE - sizeof(struct icmphdr)];
};

// Calculate checksum
unsigned short calculate_checksum(void* b, int len) {
    unsigned short *buf = b;
    unsigned int sum = 0;
    unsigned short result;
    
    // Sum all 16-bit words
    while (len > 1) {
        sum += *buf++;
        len -= 2;
    }
    
    // Add odd byte if present
    if (len == 1) {
        sum += *(unsigned char*)buf << 8;
    }
    
    // Add carry
    while (sum >> 16) {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    
    result = ~sum;
    return result;
}

double calculate_time_diff(struct timeval start, struct timeval end) {
    return (double)(end.tv_sec - start.tv_sec) * 1000.0 + 
           (double)(end.tv_usec - start.tv_usec) / 1000.0;
}

int ping_host(const char* hostname, int count) {
    struct sockaddr_in addr;
    struct hostent *host_entry;
    int sockfd;
    struct ping_packet packet;
    struct sockaddr_in r_addr;
    struct timeval start, end;
    socklen_t addr_len = sizeof(r_addr);
    
    // Create raw socket
    sockfd = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP);
    if (sockfd < 0) {
        perror("Socket creation failed (run as root)");
        return -1;
    }
    
    // Resolve hostname
    host_entry = gethostbyname(hostname);
    if (!host_entry) {
        printf("Failed to resolve hostname: %s\n", hostname);
        close(sockfd);
        return -1;
    }
    
    // Setup address
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr = *((struct in_addr*)host_entry->h_addr);
    
    printf("PING %s (%s): %d bytes of data\n", 
           hostname, inet_ntoa(addr.sin_addr), PACKET_SIZE);
    
    for (int i = 0; i < count; i++) {
        // Prepare ICMP packet
        memset(&packet, 0, sizeof(packet));
        packet.hdr.type = ICMP_ECHO;
        packet.hdr.code = 0;
        packet.hdr.un.echo.id = getpid();
        packet.hdr.un.echo.sequence = i + 1;
        
        // Add timestamp to payload
        gettimeofday(&start, NULL);
        memcpy(packet.msg, &start, sizeof(start));
        
        // Calculate checksum
        packet.hdr.checksum = 0;
        packet.hdr.checksum = calculate_checksum(&packet, sizeof(packet));
        
        // Send packet
        if (sendto(sockfd, &packet, sizeof(packet), 0, 
                   (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("Send failed");
            continue;
        }
        
        // Receive reply
        char recv_buffer[1024];
        if (recvfrom(sockfd, recv_buffer, sizeof(recv_buffer), 0, 
                     (struct sockaddr*)&r_addr, &addr_len) < 0) {
            printf("Request timeout for icmp_seq=%d\n", i + 1);
            continue;
        }
        
        gettimeofday(&end, NULL);
        
        // Parse IP header to get ICMP header
        struct iphdr *ip_hdr = (struct iphdr*)recv_buffer;
        struct icmphdr *recv_icmp = (struct icmphdr*)(recv_buffer + (ip_hdr->ihl * 4));
        
        if (recv_icmp->type == ICMP_ECHOREPLY && recv_icmp->un.echo.id == getpid()) {
            double time_taken = calculate_time_diff(start, end);
            printf("64 bytes from %s: icmp_seq=%d ttl=%d time=%.3f ms\n",
                   inet_ntoa(r_addr.sin_addr), recv_icmp->un.echo.sequence, 
                   ip_hdr->ttl, time_taken);
        }
        
        sleep(1);
    }
    
    close(sockfd);
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <hostname>\n", argv[0]);
        return 1;
    }
    
    return ping_host(argv[1], 4);
}
```

### 2. Traceroute - Path Discovery

#### Understanding Traceroute

Traceroute discovers the path packets take to reach a destination by manipulating TTL (Time To Live) values.

**Basic Traceroute Usage:**
```bash
# Basic traceroute
traceroute google.com

# UDP traceroute (default)
traceroute -U google.com

# ICMP traceroute
traceroute -I google.com

# TCP traceroute to specific port
traceroute -T -p 80 google.com

# Maximum hops
traceroute -m 15 google.com
```

#### Custom Traceroute Implementation

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/ip.h>
#include <netinet/ip_icmp.h>
#include <netinet/udp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <time.h>
#include <sys/time.h>

#define MAX_HOPS 30
#define PACKET_SIZE 60
#define BASE_PORT 33434

int traceroute_host(const char* hostname) {
    struct sockaddr_in dest_addr, from_addr;
    struct hostent *host_entry;
    int send_sock, recv_sock;
    socklen_t from_len = sizeof(from_addr);
    int ttl;
    char packet[PACKET_SIZE];
    char recv_buffer[1024];
    struct timeval start, end;
    
    // Resolve hostname
    host_entry = gethostbyname(hostname);
    if (!host_entry) {
        printf("Failed to resolve hostname: %s\n", hostname);
        return -1;
    }
    
    // Setup destination address
    memset(&dest_addr, 0, sizeof(dest_addr));
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_addr = *((struct in_addr*)host_entry->h_addr);
    dest_addr.sin_port = htons(BASE_PORT);
    
    // Create sockets
    send_sock = socket(AF_INET, SOCK_DGRAM, 0);
    recv_sock = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP);
    
    if (send_sock < 0 || recv_sock < 0) {
        perror("Socket creation failed (run as root)");
        return -1;
    }
    
    printf("traceroute to %s (%s), %d hops max, %d byte packets\n",
           hostname, inet_ntoa(dest_addr.sin_addr), MAX_HOPS, PACKET_SIZE);
    
    for (ttl = 1; ttl <= MAX_HOPS; ttl++) {
        // Set TTL for outgoing packet
        if (setsockopt(send_sock, IPPROTO_IP, IP_TTL, &ttl, sizeof(ttl)) < 0) {
            perror("setsockopt TTL failed");
            break;
        }
        
        printf("%2d  ", ttl);
        fflush(stdout);
        
        // Send three probes
        for (int probe = 0; probe < 3; probe++) {
            dest_addr.sin_port = htons(BASE_PORT + ttl);
            
            gettimeofday(&start, NULL);
            
            // Send UDP packet
            if (sendto(send_sock, packet, PACKET_SIZE, 0, 
                      (struct sockaddr*)&dest_addr, sizeof(dest_addr)) < 0) {
                perror("sendto failed");
                continue;
            }
            
            // Set receive timeout
            struct timeval timeout = {3, 0};
            if (setsockopt(recv_sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
                perror("setsockopt timeout failed");
            }
            
            // Receive ICMP response
            if (recvfrom(recv_sock, recv_buffer, sizeof(recv_buffer), 0,
                        (struct sockaddr*)&from_addr, &from_len) < 0) {
                printf(" *");
                continue;
            }
            
            gettimeofday(&end, NULL);
            
            double time_taken = (double)(end.tv_sec - start.tv_sec) * 1000.0 + 
                              (double)(end.tv_usec - start.tv_usec) / 1000.0;
            
            if (probe == 0) {
                struct hostent *hop_host = gethostbyaddr(&from_addr.sin_addr, 
                                                        sizeof(from_addr.sin_addr), AF_INET);
                if (hop_host) {
                    printf("%s (%s)  %.3f ms", hop_host->h_name, 
                           inet_ntoa(from_addr.sin_addr), time_taken);
                } else {
                    printf("%s  %.3f ms", inet_ntoa(from_addr.sin_addr), time_taken);
                }
            } else {
                printf("  %.3f ms", time_taken);
            }
        }
        
        printf("\n");
        
        // Check if we reached the destination
        if (from_addr.sin_addr.s_addr == dest_addr.sin_addr.s_addr) {
            break;
        }
    }
    
    close(send_sock);
    close(recv_sock);
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <hostname>\n", argv[0]);
        return 1;
    }
    
    return traceroute_host(argv[1]);
}
```

### 3. Netstat and SS - Connection Monitoring

#### Understanding Connection States

```bash
# Show all connections
netstat -a

# Show listening ports
netstat -l

# Show TCP connections
netstat -t

# Show UDP connections
netstat -u

# Show process information
netstat -p

# Show numerical addresses
netstat -n

# Continuous monitoring
netstat -c

# Modern replacement: ss command
ss -tuln          # TCP and UDP listening sockets
ss -tp            # TCP with process info
ss -i             # Show socket statistics
ss dst 192.168.1.1  # Connections to specific host
```

#### Custom Network Connection Monitor

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void parse_proc_net_tcp() {
    FILE *fp;
    char line[256];
    char local_addr[64], rem_addr[64], state[16];
    unsigned int local_ip, local_port, rem_ip, rem_port, st;
    
    fp = fopen("/proc/net/tcp", "r");
    if (!fp) {
        perror("Failed to open /proc/net/tcp");
        return;
    }
    
    printf("Proto Local Address          Foreign Address        State\n");
    
    // Skip header line
    fgets(line, sizeof(line), fp);
    
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "%*d: %X:%X %X:%X %X", 
                   &local_ip, &local_port, &rem_ip, &rem_port, &st) == 5) {
            
            // Convert addresses to readable format
            snprintf(local_addr, sizeof(local_addr), "%d.%d.%d.%d:%d",
                     local_ip & 0xFF, (local_ip >> 8) & 0xFF,
                     (local_ip >> 16) & 0xFF, (local_ip >> 24) & 0xFF,
                     local_port);
            
            snprintf(rem_addr, sizeof(rem_addr), "%d.%d.%d.%d:%d",
                     rem_ip & 0xFF, (rem_ip >> 8) & 0xFF,
                     (rem_ip >> 16) & 0xFF, (rem_ip >> 24) & 0xFF,
                     rem_port);
            
            // Decode state
            switch (st) {
                case 1: strcpy(state, "ESTABLISHED"); break;
                case 2: strcpy(state, "SYN_SENT"); break;
                case 3: strcpy(state, "SYN_RECV"); break;
                case 4: strcpy(state, "FIN_WAIT1"); break;
                case 5: strcpy(state, "FIN_WAIT2"); break;
                case 6: strcpy(state, "TIME_WAIT"); break;
                case 7: strcpy(state, "CLOSE"); break;
                case 8: strcpy(state, "CLOSE_WAIT"); break;
                case 9: strcpy(state, "LAST_ACK"); break;
                case 10: strcpy(state, "LISTEN"); break;
                case 11: strcpy(state, "CLOSING"); break;
                default: strcpy(state, "UNKNOWN"); break;
            }
            
            printf("tcp   %-22s %-22s %s\n", local_addr, rem_addr, state);
        }
    }
    
    fclose(fp);
}

int main() {
    parse_proc_net_tcp();
    return 0;
}
```

## Packet Capture and Analysis

### 1. TCPDump - Command Line Packet Capture

#### Basic TCPDump Usage

```bash
# Capture all traffic on interface
tcpdump -i eth0

# Capture specific number of packets
tcpdump -c 10

# Capture to file
tcpdump -w capture.pcap

# Read from file
tcpdump -r capture.pcap

# Verbose output
tcpdump -v

# More verbose with hex output
tcpdump -vvv -x

# Capture specific protocol
tcpdump tcp
tcpdump udp
tcpdump icmp

# Capture specific port
tcpdump port 80
tcpdump port 22

# Capture specific host
tcpdump host 192.168.1.1

# Complex filters
tcpdump "tcp port 80 and host google.com"
tcpdump "icmp or (tcp port 22)"
tcpdump "net 192.168.1.0/24"
```

#### Advanced TCPDump Examples

```bash
# HTTP traffic analysis
tcpdump -i any -s 0 -A 'tcp port 80 and (tcp[tcpflags] & tcp-push != 0)'

# DNS queries
tcpdump -i any -s 0 port 53

# SYN packets only
tcpdump 'tcp[tcpflags] & tcp-syn != 0'

# Monitor connection establishment
tcpdump 'tcp[tcpflags] & (tcp-syn|tcp-fin) != 0'

# Large packets (potential issues)
tcpdump 'ip[2:2] > 1500'

# Fragmented packets
tcpdump 'ip[6:2] & 0x3fff != 0'
```

#### Custom Packet Capture Implementation

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
#include <time.h>

#define BUFFER_SIZE 65536

struct packet_info {
    time_t timestamp;
    char src_ip[INET_ADDRSTRLEN];
    char dst_ip[INET_ADDRSTRLEN];
    int src_port;
    int dst_port;
    char protocol[10];
    int packet_size;
};

void print_packet_info(struct packet_info *info) {
    char time_str[26];
    strftime(time_str, 26, "%Y-%m-%d %H:%M:%S", localtime(&info->timestamp));
    
    printf("%s | %s | %s:%-5d -> %s:%-5d | Size: %d bytes\n",
           time_str, info->protocol, info->src_ip, info->src_port,
           info->dst_ip, info->dst_port, info->packet_size);
}

void parse_ip_packet(char *buffer, int size) {
    struct iphdr *ip_header = (struct iphdr*)buffer;
    struct packet_info info;
    
    // Get timestamp
    info.timestamp = time(NULL);
    info.packet_size = size;
    
    // Extract IP addresses
    struct in_addr src_addr, dst_addr;
    src_addr.s_addr = ip_header->saddr;
    dst_addr.s_addr = ip_header->daddr;
    strcpy(info.src_ip, inet_ntoa(src_addr));
    strcpy(info.dst_ip, inet_ntoa(dst_addr));
    
    // Parse based on protocol
    switch (ip_header->protocol) {
        case IPPROTO_TCP: {
            struct tcphdr *tcp_header = (struct tcphdr*)(buffer + (ip_header->ihl * 4));
            strcpy(info.protocol, "TCP");
            info.src_port = ntohs(tcp_header->source);
            info.dst_port = ntohs(tcp_header->dest);
            
            // Check TCP flags
            if (tcp_header->syn) strcat(info.protocol, " SYN");
            if (tcp_header->ack) strcat(info.protocol, " ACK");
            if (tcp_header->fin) strcat(info.protocol, " FIN");
            if (tcp_header->rst) strcat(info.protocol, " RST");
            
            break;
        }
        case IPPROTO_UDP: {
            struct udphdr *udp_header = (struct udphdr*)(buffer + (ip_header->ihl * 4));
            strcpy(info.protocol, "UDP");
            info.src_port = ntohs(udp_header->source);
            info.dst_port = ntohs(udp_header->dest);
            break;
        }
        case IPPROTO_ICMP: {
            struct icmphdr *icmp_header = (struct icmphdr*)(buffer + (ip_header->ihl * 4));
            strcpy(info.protocol, "ICMP");
            info.src_port = 0;
            info.dst_port = 0;
            
            // Add ICMP type information
            switch (icmp_header->type) {
                case ICMP_ECHOREPLY: strcat(info.protocol, " Echo Reply"); break;
                case ICMP_ECHO: strcat(info.protocol, " Echo Request"); break;
                case ICMP_DEST_UNREACH: strcat(info.protocol, " Dest Unreachable"); break;
                case ICMP_TIME_EXCEEDED: strcat(info.protocol, " Time Exceeded"); break;
            }
            break;
        }
        default:
            sprintf(info.protocol, "Protocol %d", ip_header->protocol);
            info.src_port = 0;
            info.dst_port = 0;
    }
    
    print_packet_info(&info);
}

int start_packet_capture() {
    int sock_fd;
    char buffer[BUFFER_SIZE];
    struct sockaddr saddr;
    socklen_t saddr_len = sizeof(saddr);
    
    // Create raw socket
    sock_fd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (sock_fd < 0) {
        perror("Socket creation failed (run as root)");
        return -1;
    }
    
    printf("Starting packet capture... (Press Ctrl+C to stop)\n");
    printf("Timestamp            | Proto | Source          -> Destination     | Size\n");
    printf("================================================================================\n");
    
    while (1) {
        int packet_size = recvfrom(sock_fd, buffer, BUFFER_SIZE, 0, &saddr, &saddr_len);
        if (packet_size < 0) {
            perror("Packet receive failed");
            break;
        }
        
        // Skip Ethernet header (14 bytes) and parse IP packet
        if (packet_size > 14) {
            parse_ip_packet(buffer + 14, packet_size - 14);
        }
    }
    
    close(sock_fd);
    return 0;
}

int main() {
    return start_packet_capture();
}
```

### 2. Wireshark Analysis Techniques

#### Essential Wireshark Filters

```bash
# Display filters (applied after capture)
ip.addr == 192.168.1.1              # Specific IP address
tcp.port == 80                       # HTTP traffic
dns                                  # DNS traffic only
http.request.method == "GET"         # HTTP GET requests
tcp.flags.syn == 1                   # TCP SYN packets
icmp.type == 8                       # ICMP Echo requests

# Capture filters (applied during capture)
host 192.168.1.1                     # Traffic to/from specific host
port 443                             # HTTPS traffic
not broadcast and not multicast       # Unicast only
tcp portrange 1-1024                  # Well-known ports
```

#### Protocol Analysis Examples

**TCP Connection Analysis:**
```c
// Code to demonstrate TCP connection states
#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

void demonstrate_tcp_handshake() {
    int sock_fd;
    struct sockaddr_in server_addr;
    
    // Create socket
    sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    
    // Setup server address
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(80);
    inet_pton(AF_INET, "8.8.8.8", &server_addr.sin_addr);
    
    printf("Initiating TCP connection (watch in Wireshark)...\n");
    
    // This will generate:
    // 1. SYN packet (client -> server)
    // 2. SYN-ACK packet (server -> client)  
    // 3. ACK packet (client -> server)
    if (connect(sock_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == 0) {
        printf("Connection established\n");
        
        // Send some data
        char *request = "GET / HTTP/1.1\r\nHost: 8.8.8.8\r\n\r\n";
        send(sock_fd, request, strlen(request), 0);
        
        // Close connection (FIN packets)
        close(sock_fd);
        printf("Connection closed\n");
    }
}
```

**HTTP Traffic Analysis:**
```bash
# Wireshark filters for HTTP analysis
http.request                          # All HTTP requests
http.response                         # All HTTP responses
http.request.uri contains "login"     # Login pages
http.response.code == 404            # Not found errors
http.content_length > 1000           # Large responses
```

### 3. Network Performance Analysis

#### Bandwidth Measurement

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include <sys/time.h>

#define BUFFER_SIZE 8192
#define TEST_DURATION 10

double get_time_diff(struct timeval start, struct timeval end) {
    return (double)(end.tv_sec - start.tv_sec) + 
           (double)(end.tv_usec - start.tv_usec) / 1000000.0;
}

void bandwidth_test_client(const char* server_ip, int port) {
    int sock_fd;
    struct sockaddr_in server_addr;
    char buffer[BUFFER_SIZE];
    struct timeval start, end;
    long bytes_sent = 0;
    
    // Create socket
    sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        perror("Socket creation failed");
        return;
    }
    
    // Setup server address
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    inet_pton(AF_INET, server_ip, &server_addr.sin_addr);
    
    // Connect to server
    if (connect(sock_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connection failed");
        close(sock_fd);
        return;
    }
    
    printf("Starting bandwidth test to %s:%d for %d seconds...\n", 
           server_ip, port, TEST_DURATION);
    
    // Fill buffer with test data
    memset(buffer, 'A', BUFFER_SIZE);
    
    gettimeofday(&start, NULL);
    time_t end_time = time(NULL) + TEST_DURATION;
    
    while (time(NULL) < end_time) {
        int sent = send(sock_fd, buffer, BUFFER_SIZE, 0);
        if (sent > 0) {
            bytes_sent += sent;
        } else {
            break;
        }
    }
    
    gettimeofday(&end, NULL);
    double duration = get_time_diff(start, end);
    
    double bandwidth_mbps = (bytes_sent * 8.0) / (duration * 1000000.0);
    
    printf("Test completed:\n");
    printf("  Duration: %.2f seconds\n", duration);
    printf("  Bytes sent: %ld\n", bytes_sent);
    printf("  Bandwidth: %.2f Mbps\n", bandwidth_mbps);
    
    close(sock_fd);
}

void latency_test(const char* server_ip, int count) {
    // Implementation similar to ping but measuring application-level latency
    printf("Application-level latency test to %s (%d probes):\n", server_ip, count);
    
    for (int i = 0; i < count; i++) {
        int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in server_addr;
        struct timeval start, end;
        
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(80);
        inet_pton(AF_INET, server_ip, &server_addr.sin_addr);
        
        gettimeofday(&start, NULL);
        
        if (connect(sock_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == 0) {
            gettimeofday(&end, NULL);
            double latency = get_time_diff(start, end) * 1000.0;  // Convert to ms
            printf("Probe %d: %.3f ms\n", i + 1, latency);
        } else {
            printf("Probe %d: Connection failed\n", i + 1);
        }
        
        close(sock_fd);
        usleep(100000);  // 100ms delay between probes
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <server_ip> <test_type>\n", argv[0]);
        printf("Test types: bandwidth, latency\n");
        return 1;
    }
    
    if (strcmp(argv[2], "bandwidth") == 0) {
        bandwidth_test_client(argv[1], 12345);
    } else if (strcmp(argv[2], "latency") == 0) {
        latency_test(argv[1], 10);
    } else {
        printf("Unknown test type: %s\n", argv[2]);
    }
    
    return 0;
}
```

## Advanced Network Security Analysis

### 1. Intrusion Detection and Security Monitoring

#### Port Scanning Detection

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>

#define MAX_CONNECTIONS 1000
#define TIME_WINDOW 60  // seconds

struct connection_log {
    char src_ip[INET_ADDRSTRLEN];
    int dst_port;
    time_t timestamp;
};

struct connection_log conn_log[MAX_CONNECTIONS];
int log_count = 0;

void log_connection(const char* src_ip, int dst_port) {
    if (log_count < MAX_CONNECTIONS) {
        strcpy(conn_log[log_count].src_ip, src_ip);
        conn_log[log_count].dst_port = dst_port;
        conn_log[log_count].timestamp = time(NULL);
        log_count++;
    }
}

int detect_port_scan(const char* src_ip) {
    time_t current_time = time(NULL);
    int recent_connections = 0;
    int unique_ports = 0;
    int ports[65536] = {0};
    
    for (int i = 0; i < log_count; i++) {
        if (strcmp(conn_log[i].src_ip, src_ip) == 0 && 
            (current_time - conn_log[i].timestamp) <= TIME_WINDOW) {
            recent_connections++;
            
            if (!ports[conn_log[i].dst_port]) {
                ports[conn_log[i].dst_port] = 1;
                unique_ports++;
            }
        }
    }
    
    // Port scan heuristics
    if (unique_ports > 20 && recent_connections > 50) {
        printf("ALERT: Potential port scan detected from %s\n", src_ip);
        printf("  Unique ports: %d, Total connections: %d\n", unique_ports, recent_connections);
        return 1;
    }
    
    return 0;
}

// Simple demonstration
int main() {
    // Simulate connection logs
    log_connection("192.168.1.100", 22);
    log_connection("192.168.1.100", 23);
    log_connection("192.168.1.100", 80);
    log_connection("192.168.1.100", 443);
    
    // Simulate rapid port scanning
    for (int i = 1; i <= 100; i++) {
        log_connection("192.168.1.200", i);
    }
    
    detect_port_scan("192.168.1.100");
    detect_port_scan("192.168.1.200");
    
    return 0;
}
```

#### DDoS Detection

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_IPS 10000
#define DDOS_THRESHOLD 1000  // packets per minute

struct traffic_stats {
    char ip[INET_ADDRSTRLEN];
    int packet_count;
    time_t first_seen;
    time_t last_seen;
};

struct traffic_stats traffic_log[MAX_IPS];
int ip_count = 0;

void update_traffic_stats(const char* src_ip) {
    time_t current_time = time(NULL);
    
    // Find existing IP or create new entry
    for (int i = 0; i < ip_count; i++) {
        if (strcmp(traffic_log[i].ip, src_ip) == 0) {
            traffic_log[i].packet_count++;
            traffic_log[i].last_seen = current_time;
            
            // Check for DDoS
            if (traffic_log[i].packet_count > DDOS_THRESHOLD && 
                (current_time - traffic_log[i].first_seen) <= 60) {
                printf("ALERT: Potential DDoS from %s - %d packets in last minute\n", 
                       src_ip, traffic_log[i].packet_count);
            }
            return;
        }
    }
    
    // New IP
    if (ip_count < MAX_IPS) {
        strcpy(traffic_log[ip_count].ip, src_ip);
        traffic_log[ip_count].packet_count = 1;
        traffic_log[ip_count].first_seen = current_time;
        traffic_log[ip_count].last_seen = current_time;
        ip_count++;
    }
}

void analyze_traffic_patterns() {
    printf("\nTraffic Analysis Report:\n");
    printf("========================\n");
    
    for (int i = 0; i < ip_count; i++) {
        if (traffic_log[i].packet_count > 100) {  // High volume IPs
            printf("High Volume: %s - %d packets\n", 
                   traffic_log[i].ip, traffic_log[i].packet_count);
        }
    }
}
```

### 2. Network Troubleshooting Methodologies

#### Systematic Network Problem Diagnosis

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

typedef struct {
    char description[256];
    int (*test_function)(const char* target);
    char remediation[512];
} diagnostic_test;

// Test functions
int test_dns_resolution(const char* hostname) {
    struct hostent *host = gethostbyname(hostname);
    if (host) {
        printf("✓ DNS Resolution: %s -> %s\n", hostname, inet_ntoa(*((struct in_addr*)host->h_addr)));
        return 1;
    } else {
        printf("✗ DNS Resolution failed for %s\n", hostname);
        return 0;
    }
}

int test_tcp_connectivity(const char* hostname) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;
    struct hostent *host = gethostbyname(hostname);
    
    if (!host) return 0;
    
    addr.sin_family = AF_INET;
    addr.sin_port = htons(80);
    addr.sin_addr = *((struct in_addr*)host->h_addr);
    
    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        printf("✓ TCP Connectivity: %s:80 reachable\n", hostname);
        close(sock);
        return 1;
    } else {
        printf("✗ TCP Connectivity: %s:80 unreachable\n", hostname);
        close(sock);
        return 0;
    }
}

int test_http_response(const char* hostname) {
    // Simplified HTTP test
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;
    struct hostent *host = gethostbyname(hostname);
    char request[] = "GET / HTTP/1.1\r\nHost: ";
    char full_request[512];
    char response[1024];
    
    if (!host) return 0;
    
    addr.sin_family = AF_INET;
    addr.sin_port = htons(80);
    addr.sin_addr = *((struct in_addr*)host->h_addr);
    
    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        snprintf(full_request, sizeof(full_request), "%s%s\r\n\r\n", request, hostname);
        send(sock, full_request, strlen(full_request), 0);
        
        int received = recv(sock, response, sizeof(response) - 1, 0);
        if (received > 0) {
            response[received] = '\0';
            if (strstr(response, "HTTP/1.1") || strstr(response, "HTTP/1.0")) {
                printf("✓ HTTP Response: Valid HTTP response received\n");
                close(sock);
                return 1;
            }
        }
    }
    
    printf("✗ HTTP Response: No valid HTTP response\n");
    close(sock);
    return 0;
}

// Diagnostic framework
diagnostic_test network_tests[] = {
    {"DNS Resolution Test", test_dns_resolution, "Check DNS servers, /etc/resolv.conf, firewall rules"},
    {"TCP Connectivity Test", test_tcp_connectivity, "Check routing, firewall, target service status"},
    {"HTTP Response Test", test_http_response, "Check web server status, application layer issues"}
};

void run_network_diagnostics(const char* target) {
    printf("Running Network Diagnostics for: %s\n", target);
    printf("===========================================\n");
    
    int total_tests = sizeof(network_tests) / sizeof(diagnostic_test);
    int passed_tests = 0;
    
    for (int i = 0; i < total_tests; i++) {
        printf("\n%d. %s\n", i + 1, network_tests[i].description);
        
        if (network_tests[i].test_function(target)) {
            passed_tests++;
        } else {
            printf("   Remediation: %s\n", network_tests[i].remediation);
        }
    }
    
    printf("\n===========================================\n");
    printf("Diagnostic Summary: %d/%d tests passed\n", passed_tests, total_tests);
    
    if (passed_tests == total_tests) {
        printf("✓ All tests passed - connectivity appears normal\n");
    } else {
        printf("✗ Some tests failed - check remediation suggestions\n");
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <hostname>\n", argv[0]);
        return 1;
    }
    
    run_network_diagnostics(argv[1]);
    return 0;
}
```

#### Network Performance Bottleneck Analysis

```bash
# Script for comprehensive network performance analysis
#!/bin/bash

echo "Network Performance Analysis"
echo "============================"

# Interface statistics
echo -e "\n1. Interface Statistics:"
cat /proc/net/dev | head -1
cat /proc/net/dev | grep -E "(eth|wlan|enp)"

# Connection states
echo -e "\n2. Connection State Summary:"
ss -s

# Top connections by state
echo -e "\n3. Active Connections:"
ss -tuln | head -10

# Bandwidth utilization
echo -e "\n4. Current Bandwidth Usage:"
if command -v iftop &> /dev/null; then
    timeout 5s iftop -t -s 5
else
    echo "iftop not available - install with: apt-get install iftop"
fi

# DNS performance
echo -e "\n5. DNS Resolution Time:"
for dns in 8.8.8.8 1.1.1.1 208.67.222.222; do
    echo -n "Testing $dns: "
    dig @$dns google.com | grep "Query time"
done

# Routing table
echo -e "\n6. Routing Information:"
ip route show

# ARP table (for local network analysis)
echo -e "\n7. ARP Table:"
ip neigh show
```

## Study Materials and Practice

### Recommended Reading
- **Primary:** "TCP/IP Illustrated, Volume 1" by W. Richard Stevens - Chapters on Network Diagnostics
- **Security:** "Network Security Assessment" by Chris McNab
- **Practical:** "Wireshark Network Analysis" by Laura Chappell
- **Advanced:** "The Practice of Network Security Monitoring" by Richard Bejtlich

### Essential Command Reference

#### Quick Diagnostic Commands
```bash
# Network connectivity
ping -c 4 <host>                      # Basic connectivity test
traceroute <host>                     # Path discovery
mtr <host>                           # Combined ping/traceroute
nmap -sS <host>                      # Port scan

# Interface and routing
ip addr show                         # Interface configuration
ip route show                        # Routing table
ip neigh show                        # ARP table
ethtool eth0                         # Interface details

# Connection monitoring
ss -tuln                             # All listening sockets
ss -tp                               # TCP with process info
netstat -rn                          # Routing table (legacy)
lsof -i                              # Files opened by network connections

# Packet capture
tcpdump -i any -w capture.pcap       # Capture all interfaces
tcpdump -r capture.pcap 'port 80'    # Read and filter
tshark -i eth0 -f "tcp port 443"     # Command-line Wireshark

# Performance monitoring
iftop                                # Real-time bandwidth usage
nethogs                              # Per-process bandwidth usage
vnstat                               # Network statistics
iperf3 -c <server>                   # Bandwidth testing
```

### Hands-on Labs

**Lab 1: Network Baseline Assessment**
- Capture normal network traffic for 1 hour
- Analyze protocol distribution
- Identify top talkers and protocols
- Document baseline performance metrics

**Lab 2: Troubleshooting Simulation**
- Set up common network problems (DNS issues, routing problems, firewall blocks)
- Practice systematic diagnosis approach
- Document resolution steps

**Lab 3: Security Analysis**
- Simulate port scan and detect with custom tools
- Analyze malicious traffic patterns
- Implement basic intrusion detection logic

**Lab 4: Performance Optimization**
- Identify network bottlenecks in test environment
- Measure before/after performance improvements
- Use multiple diagnostic tools for validation

### Practice Scenarios

**Scenario 1: Web Server Connectivity Issues**
```bash
# Symptoms: Users can't access website
# Your diagnostic approach:
1. Test DNS resolution: nslookup website.com
2. Test network connectivity: ping website.com  
3. Test port connectivity: telnet website.com 80
4. Analyze traffic: tcpdump -i any host website.com
5. Check server logs and firewall rules
```

**Scenario 2: Slow Network Performance**
```bash
# Symptoms: Network feels slow
# Your diagnostic approach:
1. Measure baseline: iperf3 between known hosts
2. Check interface utilization: iftop
3. Analyze packet loss: ping with statistics
4. Check for errors: ethtool -S eth0
5. Monitor connection states: ss -s
```

**Scenario 3: Intermittent Connection Drops**
```bash
# Symptoms: Connections randomly drop
# Your diagnostic approach:
1. Continuous monitoring: ping -i 0.1 gateway
2. Log analysis: tcpdump with timestamp
3. Check for patterns: analyze by time/protocol
4. Hardware diagnostics: check cables/interfaces
5. Environmental factors: interference, power
```

### Certification Paths
- **CompTIA Network+**: Network troubleshooting fundamentals
- **Wireshark Certified Network Analyst (WCNA)**: Packet analysis expertise
- **CCNA**: Cisco networking and diagnostics
- **Security+**: Network security analysis

### Tools Installation and Setup

#### Linux Environment Setup
```bash
# Essential tools installation
sudo apt-get update
sudo apt-get install -y tcpdump wireshark-qt tshark
sudo apt-get install -y nmap netcat-openbsd
sudo apt-get install -y iftop nethogs vnstat
sudo apt-get install -y mtr-tiny traceroute
sudo apt-get install -y iperf3 netperf

# Development tools
sudo apt-get install -y build-essential
sudo apt-get install -y libpcap-dev

# Wireshark permissions
sudo usermod -a -G wireshark $USER
```

#### Windows Environment
```powershell
# PowerShell network diagnostics
Test-NetConnection -ComputerName google.com -Port 80
Get-NetTCPConnection | Where-Object State -eq "Established"
Get-NetRoute
Get-NetAdapter

# Install Wireshark, Nmap for Windows
# Use Windows Subsystem for Linux (WSL) for Linux tools
```

## Next Steps

After mastering these network diagnostic and analysis techniques, proceed to:
- **Advanced Protocol Analysis**: Deep dive into specific protocols
- **Network Security Monitoring**: SIEM integration and automated analysis  
- **Software-Defined Networking**: Modern network architectures
- **Cloud Network Diagnostics**: AWS/Azure/GCP networking tools
- **Network Automation**: Scripted diagnostics and remediation

## Assessment Checklist

Before proceeding, ensure you can:

□ Capture and analyze packets using both tcpdump and Wireshark  
□ Implement custom network diagnostic tools in C  
□ Systematically troubleshoot network connectivity issues  
□ Identify and analyze network security threats  
□ Measure and optimize network performance  
□ Use advanced filtering techniques for packet analysis  
□ Correlate network issues across multiple diagnostic tools  
□ Document and communicate network problems effectively  

---

**Remember**: Network diagnostics is both an art and a science. The key to mastery is practicing with real network problems and building a systematic approach to problem-solving.

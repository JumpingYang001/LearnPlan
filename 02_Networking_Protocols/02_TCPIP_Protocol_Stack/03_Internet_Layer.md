# Internet Layer (Network Layer)

*Duration: 2 weeks*

## Overview

The Internet Layer is the core of the TCP/IP protocol suite, responsible for **packet routing** across networks. This layer handles addressing, routing, fragmentation, and delivery of packets between hosts that may be on different networks. It provides a **connectionless, best-effort delivery service**.

### Key Responsibilities
- **Logical Addressing**: Using IP addresses to identify devices
- **Routing**: Determining the best path for packets across networks
- **Fragmentation**: Breaking large packets into smaller pieces when needed
- **Error Reporting**: Using ICMP to report delivery problems
- **Quality of Service**: Managing traffic priority and flow control

## Internet Protocol Version 4 (IPv4)

### IPv4 Fundamentals

IPv4 is the most widely used version of the Internet Protocol, using **32-bit addresses** to identify devices on networks.

#### IPv4 Address Structure
```
IPv4 Address: 32 bits (4 bytes)
Format: dotted-decimal notation (e.g., 192.168.1.1)

Binary:    11000000.10101000.00000001.00000001
Decimal:   192     .168     .1       .1
```

#### IPv4 Address Classes
```
Class A: 0.0.0.0     to 127.255.255.255  (Network: 8 bits,  Host: 24 bits)
Class B: 128.0.0.0   to 191.255.255.255  (Network: 16 bits, Host: 16 bits)
Class C: 192.0.0.0   to 223.255.255.255  (Network: 24 bits, Host: 8 bits)
Class D: 224.0.0.0   to 239.255.255.255  (Multicast)
Class E: 240.0.0.0   to 255.255.255.255  (Reserved)
```

### IPv4 Header Structure in Detail

#### IPv4 Header Layout
```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|Version|  IHL  |Type of Service|          Total Length         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|         Identification        |Flags|      Fragment Offset    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Time to Live |    Protocol   |         Header Checksum       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       Source Address                          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Destination Address                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Options                    |    Padding    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### Enhanced C Example: IPv4 Header with Field Explanations
```c
#include <stdio.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <string.h>

// IPv4 header structure with proper byte ordering
struct ipv4_header {
    #if defined(__LITTLE_ENDIAN_BITFIELD)
        uint8_t ihl:4,          // Internet Header Length (4 bits)
                version:4;      // Version (4 bits) - always 4 for IPv4
    #elif defined(__BIG_ENDIAN_BITFIELD)
        uint8_t version:4,
                ihl:4;
    #else
        #error "Please fix <asm/byteorder.h>"
    #endif
    
    uint8_t tos;                // Type of Service (8 bits)
    uint16_t tot_len;           // Total Length (16 bits)
    uint16_t id;                // Identification (16 bits)
    uint16_t frag_off;          // Fragment Offset (16 bits)
    uint8_t ttl;                // Time to Live (8 bits)
    uint8_t protocol;           // Protocol (8 bits)
    uint16_t check;             // Header Checksum (16 bits)
    uint32_t saddr;             // Source Address (32 bits)
    uint32_t daddr;             // Destination Address (32 bits)
} __attribute__((packed));

// Function to print IPv4 header fields
void print_ipv4_header(struct ipv4_header *ip) {
    char src_ip[INET_ADDRSTRLEN];
    char dst_ip[INET_ADDRSTRLEN];
    
    // Convert addresses to dotted decimal notation
    inet_ntop(AF_INET, &ip->saddr, src_ip, INET_ADDRSTRLEN);
    inet_ntop(AF_INET, &ip->daddr, dst_ip, INET_ADDRSTRLEN);
    
    printf("=== IPv4 Header Analysis ===\n");
    printf("Version: %d\n", ip->version);
    printf("Header Length: %d (IHL=%d, %d bytes)\n", ip->ihl * 4, ip->ihl, ip->ihl * 4);
    printf("Type of Service: 0x%02x\n", ip->tos);
    printf("Total Length: %d bytes\n", ntohs(ip->tot_len));
    printf("Identification: 0x%04x (%d)\n", ntohs(ip->id), ntohs(ip->id));
    
    // Fragment flags analysis
    uint16_t frag = ntohs(ip->frag_off);
    printf("Flags: ");
    if (frag & 0x8000) printf("Reserved ");
    if (frag & 0x4000) printf("DF(Don't Fragment) ");
    if (frag & 0x2000) printf("MF(More Fragments) ");
    printf("\n");
    printf("Fragment Offset: %d (offset=%d bytes)\n", 
           frag & 0x1FFF, (frag & 0x1FFF) * 8);
    
    printf("Time to Live: %d\n", ip->ttl);
    printf("Protocol: %d (", ip->protocol);
    
    // Protocol interpretation
    switch (ip->protocol) {
        case 1: printf("ICMP"); break;
        case 6: printf("TCP"); break;
        case 17: printf("UDP"); break;
        case 89: printf("OSPF"); break;
        default: printf("Unknown"); break;
    }
    printf(")\n");
    
    printf("Header Checksum: 0x%04x\n", ntohs(ip->check));
    printf("Source IP: %s\n", src_ip);
    printf("Destination IP: %s\n", dst_ip);
    printf("Header Size: %zu bytes\n\n", sizeof(struct ipv4_header));
}

// Function to create and populate an IPv4 header
void create_sample_ipv4_header() {
    struct ipv4_header ip;
    memset(&ip, 0, sizeof(ip));
    
    // Fill in header fields
    ip.version = 4;                          // IPv4
    ip.ihl = 5;                             // 20 bytes (no options)
    ip.tos = 0;                             // Normal service
    ip.tot_len = htons(60);                 // 20 byte header + 40 byte payload
    ip.id = htons(12345);                   // Identification
    ip.frag_off = htons(0x4000);            // Don't Fragment flag set
    ip.ttl = 64;                            // Typical TTL value
    ip.protocol = 6;                        // TCP
    ip.check = 0;                           // Checksum (calculated separately)
    
    // Sample addresses
    inet_pton(AF_INET, "192.168.1.100", &ip.saddr);  // Source
    inet_pton(AF_INET, "8.8.8.8", &ip.daddr);        // Destination (Google DNS)
    
    print_ipv4_header(&ip);
}

int main() {
    printf("IPv4 Header Structure Analysis\n");
    printf("==============================\n\n");
    
    create_sample_ipv4_header();
    
    return 0;
}
```

#### Compilation and Execution
```bash
# Compile with network headers
gcc -o ipv4_header ipv4_header.c

# Run the program
./ipv4_header
```

### IPv4 Field Explanations

**Version (4 bits)**: Always `4` for IPv4
**IHL - Internet Header Length (4 bits)**: Length of IP header in 32-bit words (minimum 5 = 20 bytes)
**Type of Service (8 bits)**: QoS and priority information
**Total Length (16 bits)**: Total packet size including header and data (max 65,535 bytes)
**Identification (16 bits)**: Unique identifier for fragmented packets
**Flags (3 bits)**: 
- Bit 0: Reserved (must be 0)
- Bit 1: DF (Don't Fragment)
- Bit 2: MF (More Fragments)
**Fragment Offset (13 bits)**: Position of fragment in original packet
**Time to Live (8 bits)**: Maximum hops before packet is discarded
**Protocol (8 bits)**: Next layer protocol (1=ICMP, 6=TCP, 17=UDP)
**Header Checksum (16 bits)**: Error detection for header only
**Source/Destination Address (32 bits each)**: Sender and receiver IP addresses

### IPv4 Addressing and Subnetting

#### Subnet Masks and CIDR Notation

**Subnet Mask**: Determines which portion of an IP address represents the network and which represents the host.

```c
#include <stdio.h>
#include <arpa/inet.h>
#include <stdint.h>

// Structure to hold network information
typedef struct {
    uint32_t ip_address;
    uint32_t subnet_mask;
    uint32_t network_address;
    uint32_t broadcast_address;
    int prefix_length;
    int total_hosts;
    int usable_hosts;
} network_info_t;

// Calculate network information from IP and subnet mask
network_info_t calculate_network_info(const char* ip_str, const char* mask_str) {
    network_info_t info;
    
    // Convert IP and mask to binary
    inet_pton(AF_INET, ip_str, &info.ip_address);
    inet_pton(AF_INET, mask_str, &info.subnet_mask);
    
    // Calculate network address
    info.network_address = info.ip_address & info.subnet_mask;
    
    // Calculate broadcast address
    info.broadcast_address = info.network_address | (~info.subnet_mask);
    
    // Calculate prefix length (CIDR)
    info.prefix_length = 0;
    uint32_t mask = ntohl(info.subnet_mask);
    while (mask) {
        info.prefix_length += (mask & 1);
        mask >>= 1;
    }
    
    // Calculate host counts
    int host_bits = 32 - info.prefix_length;
    info.total_hosts = (1 << host_bits);
    info.usable_hosts = info.total_hosts - 2; // Subtract network and broadcast
    
    return info;
}

void print_network_info(network_info_t info) {
    char ip_str[INET_ADDRSTRLEN];
    char mask_str[INET_ADDRSTRLEN];
    char network_str[INET_ADDRSTRLEN];
    char broadcast_str[INET_ADDRSTRLEN];
    
    inet_ntop(AF_INET, &info.ip_address, ip_str, INET_ADDRSTRLEN);
    inet_ntop(AF_INET, &info.subnet_mask, mask_str, INET_ADDRSTRLEN);
    inet_ntop(AF_INET, &info.network_address, network_str, INET_ADDRSTRLEN);
    inet_ntop(AF_INET, &info.broadcast_address, broadcast_str, INET_ADDRSTRLEN);
    
    printf("=== Network Analysis ===\n");
    printf("IP Address:        %s\n", ip_str);
    printf("Subnet Mask:       %s\n", mask_str);
    printf("CIDR Notation:     %s/%d\n", network_str, info.prefix_length);
    printf("Network Address:   %s\n", network_str);
    printf("Broadcast Address: %s\n", broadcast_str);
    printf("Total Hosts:       %d\n", info.total_hosts);
    printf("Usable Hosts:      %d\n", info.usable_hosts);
    
    // Calculate first and last usable IP
    uint32_t first_host = ntohl(info.network_address) + 1;
    uint32_t last_host = ntohl(info.broadcast_address) - 1;
    
    uint32_t first_host_net = htonl(first_host);
    uint32_t last_host_net = htonl(last_host);
    
    inet_ntop(AF_INET, &first_host_net, ip_str, INET_ADDRSTRLEN);
    printf("First Usable IP:   %s\n", ip_str);
    
    inet_ntop(AF_INET, &last_host_net, ip_str, INET_ADDRSTRLEN);
    printf("Last Usable IP:    %s\n\n", ip_str);
}

// Demonstrate common subnet scenarios
void demonstrate_subnetting() {
    printf("Common Subnetting Examples\n");
    printf("==========================\n\n");
    
    // Class A private network
    printf("Class A Private Network:\n");
    network_info_t class_a = calculate_network_info("10.0.0.100", "255.0.0.0");
    print_network_info(class_a);
    
    // Class B private network
    printf("Class B Private Network:\n");
    network_info_t class_b = calculate_network_info("172.16.5.10", "255.255.0.0");
    print_network_info(class_b);
    
    // Class C private network
    printf("Class C Private Network:\n");
    network_info_t class_c = calculate_network_info("192.168.1.50", "255.255.255.0");
    print_network_info(class_c);
    
    // Subnetted Class C (/26)
    printf("Subnetted Class C (/26):\n");
    network_info_t subnet = calculate_network_info("192.168.1.50", "255.255.255.192");
    print_network_info(subnet);
}

int main() {
    demonstrate_subnetting();
    return 0;
}
```

#### Variable Length Subnet Masking (VLSM)

VLSM allows different subnet masks within the same network, enabling efficient IP address allocation.

```c
// VLSM Example: Dividing 192.168.1.0/24 network
void vlsm_example() {
    printf("VLSM Example: Dividing 192.168.1.0/24\n");
    printf("=====================================\n\n");
    
    // Requirement: 3 subnets with 60, 30, and 10 hosts respectively
    
    // Subnet 1: 60 hosts (need 6 host bits: 2^6 = 64 addresses)
    // Network: 192.168.1.0/26 (255.255.255.192)
    printf("Subnet 1 (60 hosts needed):\n");
    network_info_t subnet1 = calculate_network_info("192.168.1.0", "255.255.255.192");
    print_network_info(subnet1);
    
    // Subnet 2: 30 hosts (need 5 host bits: 2^5 = 32 addresses)
    // Network: 192.168.1.64/27 (255.255.255.224)
    printf("Subnet 2 (30 hosts needed):\n");
    network_info_t subnet2 = calculate_network_info("192.168.1.64", "255.255.255.224");
    print_network_info(subnet2);
    
    // Subnet 3: 10 hosts (need 4 host bits: 2^4 = 16 addresses)
    // Network: 192.168.1.96/28 (255.255.255.240)
    printf("Subnet 3 (10 hosts needed):\n");
    network_info_t subnet3 = calculate_network_info("192.168.1.96", "255.255.255.240");
    print_network_info(subnet3);
}
```

### IPv4 Fragmentation

When a packet is larger than the Maximum Transmission Unit (MTU) of a network link, it must be fragmented.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>

#define MTU 1500
#define IP_HEADER_SIZE 20

typedef struct fragment {
    struct ipv4_header header;
    uint8_t data[MTU - IP_HEADER_SIZE];
    int data_size;
    struct fragment* next;
} fragment_t;

// Simulate packet fragmentation
fragment_t* fragment_packet(uint8_t* original_data, int data_size, 
                           uint32_t src_ip, uint32_t dst_ip, uint16_t id) {
    
    int max_fragment_data = MTU - IP_HEADER_SIZE;
    int fragments_needed = (data_size + max_fragment_data - 1) / max_fragment_data;
    
    printf("Fragmenting packet:\n");
    printf("  Original size: %d bytes\n", data_size);
    printf("  MTU: %d bytes\n", MTU);
    printf("  Max fragment data: %d bytes\n", max_fragment_data);
    printf("  Fragments needed: %d\n\n", fragments_needed);
    
    fragment_t* first_fragment = NULL;
    fragment_t* current = NULL;
    
    for (int i = 0; i < fragments_needed; i++) {
        fragment_t* frag = malloc(sizeof(fragment_t));
        memset(frag, 0, sizeof(fragment_t));
        
        // Calculate fragment data size
        int remaining_data = data_size - (i * max_fragment_data);
        frag->data_size = (remaining_data > max_fragment_data) ? 
                         max_fragment_data : remaining_data;
        
        // Copy data
        memcpy(frag->data, original_data + (i * max_fragment_data), frag->data_size);
        
        // Set up IP header
        frag->header.version = 4;
        frag->header.ihl = 5;
        frag->header.tos = 0;
        frag->header.tot_len = htons(IP_HEADER_SIZE + frag->data_size);
        frag->header.id = htons(id);
        frag->header.ttl = 64;
        frag->header.protocol = 17; // UDP
        frag->header.saddr = src_ip;
        frag->header.daddr = dst_ip;
        
        // Set fragment flags and offset
        uint16_t flags_and_offset = (i * max_fragment_data) / 8; // Offset in 8-byte units
        
        if (i < fragments_needed - 1) {
            flags_and_offset |= 0x2000; // More Fragments (MF) flag
        }
        
        frag->header.frag_off = htons(flags_and_offset);
        frag->next = NULL;
        
        // Link fragments
        if (first_fragment == NULL) {
            first_fragment = frag;
            current = frag;
        } else {
            current->next = frag;
            current = frag;
        }
        
        // Print fragment info
        printf("Fragment %d:\n", i + 1);
        printf("  Size: %d bytes (%d header + %d data)\n", 
               IP_HEADER_SIZE + frag->data_size, IP_HEADER_SIZE, frag->data_size);
        printf("  Offset: %d bytes\n", (flags_and_offset & 0x1FFF) * 8);
        printf("  More Fragments: %s\n", (flags_and_offset & 0x2000) ? "Yes" : "No");
        printf("  Fragment data starts with: ");
        for (int j = 0; j < (frag->data_size > 8 ? 8 : frag->data_size); j++) {
            printf("%02x ", frag->data[j]);
        }
        printf("\n\n");
    }
    
    return first_fragment;
}

// Free fragment list
void free_fragments(fragment_t* fragments) {
    while (fragments) {
        fragment_t* next = fragments->next;
        free(fragments);
        fragments = next;
    }
}

int main() {
    // Create sample data larger than MTU
    int data_size = 4000; // 4KB of data
    uint8_t* original_data = malloc(data_size);
    
    // Fill with sample pattern
    for (int i = 0; i < data_size; i++) {
        original_data[i] = i % 256;
    }
    
    uint32_t src_ip, dst_ip;
    inet_pton(AF_INET, "192.168.1.100", &src_ip);
    inet_pton(AF_INET, "192.168.1.200", &dst_ip);
    
    printf("IPv4 Fragmentation Example\n");
    printf("==========================\n\n");
    
    fragment_t* fragments = fragment_packet(original_data, data_size, 
                                          src_ip, dst_ip, 12345);
    
    free_fragments(fragments);
    free(original_data);
    
    return 0;
}
```

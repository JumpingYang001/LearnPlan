# Network Access Layer (Data Link Layer)

*Duration: 2 weeks*

## Overview

The **Network Access Layer** (also known as the Data Link Layer or Layer 2) is the foundation of network communication, responsible for:
- **Physical addressing** using MAC addresses
- **Frame formatting** and error detection
- **Media access control** for shared network segments
- **Network interface management** and device driver interaction

This layer bridges the gap between the physical transmission medium and higher-level network protocols, ensuring reliable data delivery between directly connected network nodes.

## Core Concepts to Master

### Network Access Layer Architecture

```
┌─────────────────────────────────────────┐
│            Network Layer                │
│         (IP, ICMP, IGMP)               │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Network Access Layer           │
│  ┌─────────────────────────────────┐   │
│  │      Logical Link Control       │   │  ← LLC Sublayer
│  │         (LLC 802.2)            │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │    Media Access Control        │   │  ← MAC Sublayer
│  │    (Ethernet, WiFi, etc.)      │   │
│  └─────────────────────────────────┘   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│           Physical Layer                │
│     (Cables, Radio, Electrical)        │
└─────────────────────────────────────────┘
```

## Ethernet Protocol Deep Dive

### Ethernet Frame Format

Ethernet is the most common LAN technology. Understanding its frame structure is crucial for network programming.

```
Ethernet Frame Structure (IEEE 802.3):
┌─────────────┬─────────────┬─────────┬─────────────┬─────┬─────┐
│   Preamble  │     SFD     │   DA    │     SA      │ Len │ FCS │
│   7 bytes   │   1 byte    │ 6 bytes │   6 bytes   │ 2 B │ 4 B │
└─────────────┴─────────────┴─────────┴─────────────┴─────┴─────┘
                            │◄──── MAC Header ────►│
                            
Ethernet II Frame Structure (DIX):
┌─────────────┬─────────────┬─────────┬─────────────┬──────┬─────┐
│   Preamble  │     SFD     │   DA    │     SA      │ Type │ FCS │
│   7 bytes   │   1 byte    │ 6 bytes │   6 bytes   │ 2 B  │ 4 B │
└─────────────┴─────────────┴─────────┴─────────────┴──────┴─────┘
```

### Enhanced Ethernet Frame Implementation

```c
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <arpa/inet.h>

// Ethernet header structure
typedef struct {
    uint8_t  dest_mac[6];       // Destination MAC address
    uint8_t  src_mac[6];        // Source MAC address
    uint16_t ethertype;         // EtherType or Length
} __attribute__((packed)) ethernet_header_t;

// Complete Ethernet frame structure
typedef struct {
    ethernet_header_t header;
    uint8_t payload[1500];      // Maximum payload size
    uint32_t fcs;              // Frame Check Sequence (CRC-32)
} __attribute__((packed)) ethernet_frame_t;

// Common EtherType values
#define ETHERTYPE_IPV4    0x0800
#define ETHERTYPE_ARP     0x0806
#define ETHERTYPE_IPV6    0x86DD
#define ETHERTYPE_VLAN    0x8100

// Function to print MAC address in human-readable format
void print_mac_address(const uint8_t mac[6]) {
    printf("%02x:%02x:%02x:%02x:%02x:%02x", 
           mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

// Function to parse Ethernet frame
void parse_ethernet_frame(const uint8_t* frame_data, size_t frame_len) {
    if (frame_len < sizeof(ethernet_header_t)) {
        printf("Frame too short for Ethernet header\n");
        return;
    }
    
    ethernet_header_t* eth = (ethernet_header_t*)frame_data;
    
    printf("=== Ethernet Frame Analysis ===\n");
    printf("Destination MAC: ");
    print_mac_address(eth->dest_mac);
    printf("\n");
    
    printf("Source MAC: ");
    print_mac_address(eth->src_mac);
    printf("\n");
    
    uint16_t ethertype = ntohs(eth->ethertype);
    printf("EtherType/Length: 0x%04x ", ethertype);
    
    if (ethertype > 1500) {
        // EtherType field
        switch (ethertype) {
            case ETHERTYPE_IPV4:
                printf("(IPv4)\n");
                break;
            case ETHERTYPE_ARP:
                printf("(ARP)\n");
                break;
            case ETHERTYPE_IPV6:
                printf("(IPv6)\n");
                break;
            case ETHERTYPE_VLAN:
                printf("(VLAN Tagged)\n");
                break;
            default:
                printf("(Unknown/Other)\n");
                break;
        }
    } else {
        // Length field (IEEE 802.3)
        printf("(Length: %d bytes)\n", ethertype);
    }
    
    printf("Payload length: %zu bytes\n", frame_len - sizeof(ethernet_header_t));
    printf("================================\n\n");
}

// Function to create an Ethernet frame
size_t create_ethernet_frame(uint8_t* buffer, size_t buffer_size,
                           const uint8_t dest_mac[6], 
                           const uint8_t src_mac[6],
                           uint16_t ethertype,
                           const uint8_t* payload,
                           size_t payload_len) {
    
    size_t frame_size = sizeof(ethernet_header_t) + payload_len;
    
    if (frame_size > buffer_size) {
        printf("Buffer too small for frame\n");
        return 0;
    }
    
    if (payload_len > 1500) {
        printf("Payload too large (max 1500 bytes)\n");
        return 0;
    }
    
    ethernet_header_t* eth = (ethernet_header_t*)buffer;
    
    // Fill Ethernet header
    memcpy(eth->dest_mac, dest_mac, 6);
    memcpy(eth->src_mac, src_mac, 6);
    eth->ethertype = htons(ethertype);
    
    // Copy payload
    if (payload && payload_len > 0) {
        memcpy(buffer + sizeof(ethernet_header_t), payload, payload_len);
    }
    
    return frame_size;
}

// Example usage
int main() {
    printf("=== Ethernet Frame Structure Demo ===\n\n");
    
    // Example MAC addresses
    uint8_t dest_mac[] = {0x00, 0x1A, 0x2B, 0x3C, 0x4D, 0x5E};
    uint8_t src_mac[]  = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
    
    // Create a sample frame
    uint8_t frame_buffer[1518];  // Max Ethernet frame size
    const char* payload = "Hello, Network World!";
    
    size_t frame_len = create_ethernet_frame(
        frame_buffer, sizeof(frame_buffer),
        dest_mac, src_mac,
        ETHERTYPE_IPV4,
        (const uint8_t*)payload, strlen(payload)
    );
    
    if (frame_len > 0) {
        printf("Created Ethernet frame (%zu bytes)\n\n", frame_len);
        parse_ethernet_frame(frame_buffer, frame_len);
        
        // Hex dump of frame header
        printf("Frame header hex dump:\n");
        for (int i = 0; i < sizeof(ethernet_header_t); i++) {
            printf("%02x ", frame_buffer[i]);
            if ((i + 1) % 8 == 0) printf("\n");
        }
        printf("\n");
    }
    
    return 0;
}
```

### MAC Address Understanding

**MAC (Media Access Control) Address** is a unique 48-bit identifier assigned to network interfaces.

```c
#include <stdio.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    uint8_t octets[6];
} mac_address_t;

// MAC address analysis functions
int is_broadcast_mac(const mac_address_t* mac) {
    for (int i = 0; i < 6; i++) {
        if (mac->octets[i] != 0xFF) return 0;
    }
    return 1;
}

int is_multicast_mac(const mac_address_t* mac) {
    return (mac->octets[0] & 0x01) != 0;
}

int is_unicast_mac(const mac_address_t* mac) {
    return !is_multicast_mac(mac) && !is_broadcast_mac(mac);
}

int is_locally_administered(const mac_address_t* mac) {
    return (mac->octets[0] & 0x02) != 0;
}

void get_oui(const mac_address_t* mac, char* oui_str) {
    sprintf(oui_str, "%02X:%02X:%02X", 
            mac->octets[0], mac->octets[1], mac->octets[2]);
}

void analyze_mac_address(const mac_address_t* mac) {
    printf("MAC Address Analysis:\n");
    printf("Address: %02X:%02X:%02X:%02X:%02X:%02X\n",
           mac->octets[0], mac->octets[1], mac->octets[2],
           mac->octets[3], mac->octets[4], mac->octets[5]);
    
    char oui[9];
    get_oui(mac, oui);
    printf("OUI (Organizationally Unique Identifier): %s\n", oui);
    
    printf("Type: ");
    if (is_broadcast_mac(mac)) {
        printf("Broadcast\n");
    } else if (is_multicast_mac(mac)) {
        printf("Multicast\n");
    } else {
        printf("Unicast\n");
    }
    
    printf("Administration: %s\n", 
           is_locally_administered(mac) ? "Locally Administered" : "Globally Unique");
    
    printf("\n");
}

// Example: Common MAC addresses
int main() {
    mac_address_t examples[] = {
        {{0x00, 0x1A, 0x2B, 0x3C, 0x4D, 0x5E}},  // Normal unicast
        {{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}},  // Broadcast
        {{0x01, 0x00, 0x5E, 0x12, 0x34, 0x56}},  // Multicast
        {{0x02, 0x1A, 0x2B, 0x3C, 0x4D, 0x5E}},  // Locally administered
    };
    
    for (int i = 0; i < 4; i++) {
        analyze_mac_address(&examples[i]);
    }
    
    return 0;
}
```

## Address Resolution Protocol (ARP)

ARP is responsible for mapping IP addresses to MAC addresses on local network segments. Understanding ARP is crucial for network troubleshooting and security.

### ARP Protocol Operation

```
ARP Request/Reply Process:
┌─────────────┐                           ┌─────────────┐
│   Host A    │                           │   Host B    │
│ IP: 192.168.1.10                       │ IP: 192.168.1.20
│ MAC: AA:BB:CC:DD:EE:FF                  │ MAC: 11:22:33:44:55:66
└─────────────┘                           └─────────────┘
       │                                         │
       │ 1. ARP Request (Broadcast)             │
       │ "Who has 192.168.1.20?"                │
       ├─────────────────────────────────────────┤
       │                                         │
       │ 2. ARP Reply (Unicast)                  │
       │ "192.168.1.20 is at 11:22:33:44:55:66" │
       │◄────────────────────────────────────────┤
       │                                         │
```

### Enhanced ARP Implementation

```c
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/if_ether.h>

// ARP packet structure (RFC 826)
typedef struct {
    uint16_t htype;         // Hardware type (1 = Ethernet)
    uint16_t ptype;         // Protocol type (0x0800 = IPv4)
    uint8_t  hlen;          // Hardware address length (6 for Ethernet)
    uint8_t  plen;          // Protocol address length (4 for IPv4)
    uint16_t oper;          // Operation (1 = request, 2 = reply)
    uint8_t  sha[6];        // Sender hardware address
    uint8_t  spa[4];        // Sender protocol address
    uint8_t  tha[6];        // Target hardware address
    uint8_t  tpa[4];        // Target protocol address
} __attribute__((packed)) arp_packet_t;

// ARP operation codes
#define ARP_REQUEST  1
#define ARP_REPLY    2

// Hardware and protocol types
#define HTYPE_ETHERNET  1
#define PTYPE_IPV4      0x0800

// Utility functions
void print_ip_address(const uint8_t ip[4]) {
    printf("%d.%d.%d.%d", ip[0], ip[1], ip[2], ip[3]);
}

void print_mac_address(const uint8_t mac[6]) {
    printf("%02x:%02x:%02x:%02x:%02x:%02x", 
           mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

// Convert IP string to byte array
void ip_string_to_bytes(const char* ip_str, uint8_t ip[4]) {
    struct in_addr addr;
    inet_aton(ip_str, &addr);
    memcpy(ip, &addr.s_addr, 4);
}

// Parse ARP packet
void parse_arp_packet(const arp_packet_t* arp) {
    printf("=== ARP Packet Analysis ===\n");
    printf("Hardware Type: %d ", ntohs(arp->htype));
    if (ntohs(arp->htype) == HTYPE_ETHERNET) {
        printf("(Ethernet)\n");
    } else {
        printf("(Other)\n");
    }
    
    printf("Protocol Type: 0x%04x ", ntohs(arp->ptype));
    if (ntohs(arp->ptype) == PTYPE_IPV4) {
        printf("(IPv4)\n");
    } else {
        printf("(Other)\n");
    }
    
    printf("Hardware Address Length: %d bytes\n", arp->hlen);
    printf("Protocol Address Length: %d bytes\n", arp->plen);
    
    printf("Operation: %d ", ntohs(arp->oper));
    switch (ntohs(arp->oper)) {
        case ARP_REQUEST:
            printf("(ARP Request)\n");
            break;
        case ARP_REPLY:
            printf("(ARP Reply)\n");
            break;
        default:
            printf("(Unknown)\n");
    }
    
    printf("Sender Hardware Address: ");
    print_mac_address(arp->sha);
    printf("\n");
    
    printf("Sender Protocol Address: ");
    print_ip_address(arp->spa);
    printf("\n");
    
    printf("Target Hardware Address: ");
    print_mac_address(arp->tha);
    printf("\n");
    
    printf("Target Protocol Address: ");
    print_ip_address(arp->tpa);
    printf("\n");
    printf("===========================\n\n");
}

// Create ARP request packet
void create_arp_request(arp_packet_t* arp,
                       const uint8_t sender_mac[6],
                       const char* sender_ip_str,
                       const char* target_ip_str) {
    memset(arp, 0, sizeof(arp_packet_t));
    
    arp->htype = htons(HTYPE_ETHERNET);
    arp->ptype = htons(PTYPE_IPV4);
    arp->hlen = 6;
    arp->plen = 4;
    arp->oper = htons(ARP_REQUEST);
    
    // Set sender addresses
    memcpy(arp->sha, sender_mac, 6);
    ip_string_to_bytes(sender_ip_str, arp->spa);
    
    // Target MAC is unknown (set to zero for request)
    memset(arp->tha, 0, 6);
    ip_string_to_bytes(target_ip_str, arp->tpa);
}

// Create ARP reply packet
void create_arp_reply(arp_packet_t* arp,
                     const uint8_t sender_mac[6],
                     const char* sender_ip_str,
                     const uint8_t target_mac[6],
                     const char* target_ip_str) {
    memset(arp, 0, sizeof(arp_packet_t));
    
    arp->htype = htons(HTYPE_ETHERNET);
    arp->ptype = htons(PTYPE_IPV4);
    arp->hlen = 6;
    arp->plen = 4;
    arp->oper = htons(ARP_REPLY);
    
    // Set addresses
    memcpy(arp->sha, sender_mac, 6);
    ip_string_to_bytes(sender_ip_str, arp->spa);
    memcpy(arp->tha, target_mac, 6);
    ip_string_to_bytes(target_ip_str, arp->tpa);
}

// ARP cache entry structure
typedef struct {
    uint8_t ip[4];
    uint8_t mac[6];
    time_t timestamp;
    int is_static;
} arp_cache_entry_t;

#define MAX_ARP_ENTRIES 256

typedef struct {
    arp_cache_entry_t entries[MAX_ARP_ENTRIES];
    int count;
} arp_cache_t;

// Simple ARP cache implementation
arp_cache_t arp_cache = {0};

int add_arp_entry(const uint8_t ip[4], const uint8_t mac[6]) {
    if (arp_cache.count >= MAX_ARP_ENTRIES) {
        printf("ARP cache full\n");
        return -1;
    }
    
    arp_cache_entry_t* entry = &arp_cache.entries[arp_cache.count];
    memcpy(entry->ip, ip, 4);
    memcpy(entry->mac, mac, 6);
    entry->timestamp = time(NULL);
    entry->is_static = 0;
    
    arp_cache.count++;
    printf("Added ARP entry: ");
    print_ip_address(ip);
    printf(" -> ");
    print_mac_address(mac);
    printf("\n");
    
    return 0;
}

void print_arp_cache() {
    printf("=== ARP Cache ===\n");
    printf("IP Address       MAC Address         Age     Type\n");
    printf("------------------------------------------------\n");
    
    time_t now = time(NULL);
    for (int i = 0; i < arp_cache.count; i++) {
        arp_cache_entry_t* entry = &arp_cache.entries[i];
        
        print_ip_address(entry->ip);
        printf("    ");
        print_mac_address(entry->mac);
        printf("  %3lds   %s\n",
               now - entry->timestamp,
               entry->is_static ? "Static" : "Dynamic");
    }
    printf("=================\n\n");
}

// Example usage
int main() {
    printf("=== ARP Protocol Demonstration ===\n\n");
    
    // Example MAC addresses
    uint8_t host_a_mac[] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
    uint8_t host_b_mac[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    
    // Create ARP request
    arp_packet_t arp_req;
    create_arp_request(&arp_req, host_a_mac, "192.168.1.10", "192.168.1.20");
    
    printf("1. ARP Request created:\n");
    parse_arp_packet(&arp_req);
    
    // Create ARP reply
    arp_packet_t arp_reply;
    create_arp_reply(&arp_reply, host_b_mac, "192.168.1.20", 
                     host_a_mac, "192.168.1.10");
    
    printf("2. ARP Reply created:\n");
    parse_arp_packet(&arp_reply);
    
    // Simulate ARP cache operations
    printf("3. Updating ARP cache:\n");
    add_arp_entry(arp_reply.spa, arp_reply.sha);
    print_arp_cache();
    
    // Show packet sizes
    printf("ARP packet size: %zu bytes\n", sizeof(arp_packet_t));
    printf("Ethernet + ARP total: %zu bytes\n", 
           sizeof(ethernet_header_t) + sizeof(arp_packet_t));
    
    return 0;
}
```

### ARP Security Considerations

ARP is inherently insecure and vulnerable to various attacks:

```c
// Example: ARP Spoofing Detection
#include <stdio.h>
#include <time.h>

typedef struct {
    uint8_t ip[4];
    uint8_t mac[6];
    time_t last_seen;
    int change_count;
} arp_monitor_entry_t;

#define MAX_MONITOR_ENTRIES 100
arp_monitor_entry_t monitor_table[MAX_MONITOR_ENTRIES];
int monitor_count = 0;

// Monitor ARP packets for suspicious activity
void monitor_arp_packet(const arp_packet_t* arp) {
    if (ntohs(arp->oper) != ARP_REPLY) return;
    
    time_t now = time(NULL);
    
    // Look for existing entry
    for (int i = 0; i < monitor_count; i++) {
        if (memcmp(monitor_table[i].ip, arp->spa, 4) == 0) {
            // Check if MAC address changed
            if (memcmp(monitor_table[i].mac, arp->sha, 6) != 0) {
                printf("⚠️  ARP SPOOFING DETECTED!\n");
                printf("IP: ");
                print_ip_address(arp->spa);
                printf("\nOld MAC: ");
                print_mac_address(monitor_table[i].mac);
                printf("\nNew MAC: ");
                print_mac_address(arp->sha);
                printf("\nTime since last change: %ld seconds\n\n",
                       now - monitor_table[i].last_seen);
                
                monitor_table[i].change_count++;
                memcpy(monitor_table[i].mac, arp->sha, 6);
            }
            monitor_table[i].last_seen = now;
            return;
        }
    }
    
    // Add new entry
    if (monitor_count < MAX_MONITOR_ENTRIES) {
        memcpy(monitor_table[monitor_count].ip, arp->spa, 4);
        memcpy(monitor_table[monitor_count].mac, arp->sha, 6);
        monitor_table[monitor_count].last_seen = now;
        monitor_table[monitor_count].change_count = 0;
        monitor_count++;
    }
}
```

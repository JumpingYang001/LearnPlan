# Network Protocol Analyzer Project

*Last Updated: June 21, 2025*

## Project Overview

This project implements a comprehensive network protocol analyzer (packet sniffer) that captures, analyzes, and displays network traffic in real-time. The analyzer demonstrates advanced socket programming concepts including raw sockets, packet parsing, protocol dissection, and network monitoring techniques.

## Learning Objectives

- Understand raw socket programming and packet capture
- Learn network protocol structures and parsing
- Implement packet filtering and analysis
- Practice network monitoring and troubleshooting
- Explore security aspects of network analysis
- Develop real-time data processing skills

## Architecture Overview

```
network_protocol_analyzer/
├── src/
│   ├── main.c                      # Main application entry point
│   ├── packet_capture.c            # Raw socket packet capture
│   ├── protocol_parsers.c          # Protocol parsing functions
│   ├── packet_filter.c             # Packet filtering engine
│   ├── analyzer_engine.c           # Analysis and statistics
│   ├── display_manager.c           # Real-time display management
│   ├── config_manager.c            # Configuration management
│   ├── log_manager.c               # Logging and output
│   └── utils.c                     # Utility functions
├── include/
│   ├── packet_capture.h
│   ├── protocol_parsers.h
│   ├── packet_filter.h
│   ├── analyzer_engine.h
│   ├── display_manager.h
│   ├── config_manager.h
│   ├── log_manager.h
│   ├── common.h
│   └── protocols.h
├── filters/
│   ├── default.filter              # Default packet filters
│   ├── http.filter                 # HTTP-specific filters
│   └── security.filter             # Security-focused filters
├── configs/
│   └── analyzer.conf               # Analyzer configuration
├── scripts/
│   ├── setup_permissions.sh        # Setup script for raw sockets
│   └── generate_traffic.py         # Traffic generation for testing
├── Makefile
└── README.md
```

## Core Headers

### Common Definitions (include/common.h)

```c
#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <netinet/ip_icmp.h>
#include <netinet/if_ether.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <pthread.h>
#include <pcap/pcap.h>
#include <ncurses.h>

// Buffer sizes
#define MAX_PACKET_SIZE 65536
#define MAX_INTERFACE_NAME 16
#define MAX_FILTER_LENGTH 256
#define MAX_PROTOCOL_NAME 32
#define MAX_ADDRESS_LENGTH 64

// Display settings
#define MAX_DISPLAY_PACKETS 1000
#define REFRESH_INTERVAL_MS 100
#define STATS_UPDATE_INTERVAL 5

// Protocol identifiers
#define PROTOCOL_IP 0x0800
#define PROTOCOL_ARP 0x0806
#define PROTOCOL_IPV6 0x86DD

// Packet directions
typedef enum {
    PACKET_DIRECTION_UNKNOWN = 0,
    PACKET_DIRECTION_INGRESS,
    PACKET_DIRECTION_EGRESS,
    PACKET_DIRECTION_LOCAL
} packet_direction_t;

// Packet types
typedef enum {
    PACKET_TYPE_UNKNOWN = 0,
    PACKET_TYPE_ETHERNET,
    PACKET_TYPE_IP,
    PACKET_TYPE_TCP,
    PACKET_TYPE_UDP,
    PACKET_TYPE_ICMP,
    PACKET_TYPE_ARP,
    PACKET_TYPE_HTTP,
    PACKET_TYPE_DNS,
    PACKET_TYPE_DHCP
} packet_type_t;

// Global flags
extern volatile sig_atomic_t g_running;
extern int g_debug_mode;
extern int g_verbose_mode;

// Utility macros
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#endif // COMMON_H
```

### Protocol Structures (include/protocols.h)

```c
#ifndef PROTOCOLS_H
#define PROTOCOLS_H

#include "common.h"

// Ethernet frame structure
typedef struct {
    uint8_t dest_mac[6];
    uint8_t src_mac[6];
    uint16_t ethertype;
} __attribute__((packed)) ethernet_header_t;

// IP header structure
typedef struct {
    uint8_t version_ihl;
    uint8_t tos;
    uint16_t total_length;
    uint16_t identification;
    uint16_t flags_fragment;
    uint8_t ttl;
    uint8_t protocol;
    uint16_t checksum;
    uint32_t src_ip;
    uint32_t dest_ip;
} __attribute__((packed)) ip_header_t;

// TCP header structure
typedef struct {
    uint16_t src_port;
    uint16_t dest_port;
    uint32_t sequence;
    uint32_t acknowledgment;
    uint8_t data_offset_flags;
    uint8_t flags;
    uint16_t window;
    uint16_t checksum;
    uint16_t urgent_pointer;
} __attribute__((packed)) tcp_header_t;

// UDP header structure
typedef struct {
    uint16_t src_port;
    uint16_t dest_port;
    uint16_t length;
    uint16_t checksum;
} __attribute__((packed)) udp_header_t;

// ICMP header structure
typedef struct {
    uint8_t type;
    uint8_t code;
    uint16_t checksum;
    uint32_t rest;
} __attribute__((packed)) icmp_header_t;

// ARP header structure
typedef struct {
    uint16_t hardware_type;
    uint16_t protocol_type;
    uint8_t hardware_size;
    uint8_t protocol_size;
    uint16_t opcode;
    uint8_t sender_mac[6];
    uint32_t sender_ip;
    uint8_t target_mac[6];
    uint32_t target_ip;
} __attribute__((packed)) arp_header_t;

// DNS header structure
typedef struct {
    uint16_t id;
    uint16_t flags;
    uint16_t qdcount;
    uint16_t ancount;
    uint16_t nscount;
    uint16_t arcount;
} __attribute__((packed)) dns_header_t;

// HTTP method types
typedef enum {
    HTTP_METHOD_GET = 0,
    HTTP_METHOD_POST,
    HTTP_METHOD_PUT,
    HTTP_METHOD_DELETE,
    HTTP_METHOD_HEAD,
    HTTP_METHOD_OPTIONS,
    HTTP_METHOD_PATCH,
    HTTP_METHOD_UNKNOWN
} http_method_t;

// Parsed packet information
typedef struct parsed_packet {
    // Timing information
    struct timeval timestamp;
    
    // Basic packet info
    size_t packet_length;
    packet_type_t packet_type;
    packet_direction_t direction;
    
    // Ethernet layer
    ethernet_header_t* eth_header;
    
    // Network layer
    ip_header_t* ip_header;
    char src_ip_str[INET_ADDRSTRLEN];
    char dest_ip_str[INET_ADDRSTRLEN];
    
    // Transport layer
    union {
        tcp_header_t* tcp_header;
        udp_header_t* udp_header;
        icmp_header_t* icmp_header;
    };
    uint16_t src_port;
    uint16_t dest_port;
    
    // Application layer
    char* payload;
    size_t payload_length;
    
    // Protocol-specific information
    union {
        struct {
            http_method_t method;
            char* url;
            char* user_agent;
            int status_code;
        } http;
        
        struct {
            uint16_t query_id;
            char* domain;
            uint16_t query_type;
        } dns;
        
        struct {
            uint16_t opcode;
            char sender_mac_str[18];
            char sender_ip_str[INET_ADDRSTRLEN];
        } arp;
    } app_data;
    
    // Analysis results
    int is_suspicious;
    char* analysis_notes;
    
    // Linked list for packet queue
    struct parsed_packet* next;
} parsed_packet_t;

// Function prototypes
const char* protocol_to_string(uint8_t protocol);
const char* packet_type_to_string(packet_type_t type);
const char* http_method_to_string(http_method_t method);
void free_parsed_packet(parsed_packet_t* packet);

#endif // PROTOCOLS_H
```

### Packet Capture (include/packet_capture.h)

```c
#ifndef PACKET_CAPTURE_H
#define PACKET_CAPTURE_H

#include "common.h"
#include "protocols.h"

typedef struct capture_stats {
    uint64_t total_packets;
    uint64_t total_bytes;
    uint64_t packets_per_second;
    uint64_t bytes_per_second;
    uint64_t dropped_packets;
    uint64_t error_packets;
    
    // Protocol statistics
    uint64_t tcp_packets;
    uint64_t udp_packets;
    uint64_t icmp_packets;
    uint64_t arp_packets;
    uint64_t other_packets;
    
    // Application protocol statistics
    uint64_t http_packets;
    uint64_t https_packets;
    uint64_t dns_packets;
    uint64_t dhcp_packets;
    
    // Timing
    struct timeval start_time;
    struct timeval last_update;
    
    pthread_mutex_t mutex;
} capture_stats_t;

typedef struct capture_config {
    char interface[MAX_INTERFACE_NAME];
    char filter_expression[MAX_FILTER_LENGTH];
    int promiscuous_mode;
    int buffer_size;
    int timeout_ms;
    int max_packet_size;
    int capture_limit;
    
    // Callback functions
    void (*packet_handler)(const uint8_t* packet, size_t length, 
                          const struct timeval* timestamp);
    void (*error_handler)(const char* error_msg);
} capture_config_t;

typedef struct packet_capture {
    pcap_t* pcap_handle;
    capture_config_t config;
    capture_stats_t stats;
    
    pthread_t capture_thread;
    volatile int running;
    volatile int paused;
    
    // Packet queue
    parsed_packet_t* packet_queue_head;
    parsed_packet_t* packet_queue_tail;
    int queue_size;
    int max_queue_size;
    pthread_mutex_t queue_mutex;
    pthread_cond_t queue_cond;
} packet_capture_t;

// Function prototypes
packet_capture_t* create_packet_capture(const capture_config_t* config);
void destroy_packet_capture(packet_capture_t* capture);
int start_packet_capture(packet_capture_t* capture);
int stop_packet_capture(packet_capture_t* capture);
int pause_packet_capture(packet_capture_t* capture);
int resume_packet_capture(packet_capture_t* capture);
int set_capture_filter(packet_capture_t* capture, const char* filter);
capture_stats_t get_capture_stats(packet_capture_t* capture);
int list_network_interfaces(char interfaces[][MAX_INTERFACE_NAME], int max_count);
int validate_interface(const char* interface);

#endif // PACKET_CAPTURE_H
```

### Protocol Parsers (include/protocol_parsers.h)

```c
#ifndef PROTOCOL_PARSERS_H
#define PROTOCOL_PARSERS_H

#include "common.h"
#include "protocols.h"

// Parser function types
typedef int (*ethernet_parser_t)(const uint8_t* packet, size_t length, 
                                parsed_packet_t* parsed);
typedef int (*ip_parser_t)(const uint8_t* packet, size_t length, 
                          parsed_packet_t* parsed);
typedef int (*tcp_parser_t)(const uint8_t* packet, size_t length, 
                           parsed_packet_t* parsed);
typedef int (*udp_parser_t)(const uint8_t* packet, size_t length, 
                           parsed_packet_t* parsed);
typedef int (*icmp_parser_t)(const uint8_t* packet, size_t length, 
                            parsed_packet_t* parsed);
typedef int (*arp_parser_t)(const uint8_t* packet, size_t length, 
                           parsed_packet_t* parsed);
typedef int (*http_parser_t)(const uint8_t* payload, size_t length, 
                            parsed_packet_t* parsed);
typedef int (*dns_parser_t)(const uint8_t* payload, size_t length, 
                           parsed_packet_t* parsed);

// Parser registry
typedef struct protocol_parser {
    const char* name;
    packet_type_t type;
    uint16_t protocol_id;
    union {
        ethernet_parser_t ethernet_parser;
        ip_parser_t ip_parser;
        tcp_parser_t tcp_parser;
        udp_parser_t udp_parser;
        icmp_parser_t icmp_parser;
        arp_parser_t arp_parser;
        http_parser_t http_parser;
        dns_parser_t dns_parser;
    } parser;
} protocol_parser_t;

// Main parsing functions
parsed_packet_t* parse_packet(const uint8_t* packet, size_t length, 
                             const struct timeval* timestamp);
int parse_ethernet_frame(const uint8_t* packet, size_t length, 
                        parsed_packet_t* parsed);
int parse_ip_packet(const uint8_t* packet, size_t length, 
                   parsed_packet_t* parsed);
int parse_tcp_segment(const uint8_t* packet, size_t length, 
                     parsed_packet_t* parsed);
int parse_udp_datagram(const uint8_t* packet, size_t length, 
                      parsed_packet_t* parsed);
int parse_icmp_packet(const uint8_t* packet, size_t length, 
                     parsed_packet_t* parsed);
int parse_arp_packet(const uint8_t* packet, size_t length, 
                    parsed_packet_t* parsed);

// Application layer parsers
int parse_http_message(const uint8_t* payload, size_t length, 
                      parsed_packet_t* parsed);
int parse_dns_message(const uint8_t* payload, size_t length, 
                     parsed_packet_t* parsed);
int parse_dhcp_message(const uint8_t* payload, size_t length, 
                      parsed_packet_t* parsed);

// Utility functions
int is_http_request(const uint8_t* payload, size_t length);
int is_http_response(const uint8_t* payload, size_t length);
http_method_t get_http_method(const char* method_str);
char* extract_http_header(const char* headers, const char* header_name);
uint16_t calculate_ip_checksum(const ip_header_t* ip_header);
uint16_t calculate_tcp_checksum(const ip_header_t* ip_header, 
                               const tcp_header_t* tcp_header, 
                               const uint8_t* data, size_t data_len);

// Protocol detection
packet_type_t detect_protocol_type(const uint8_t* packet, size_t length);
int identify_application_protocol(const parsed_packet_t* packet);

#endif // PROTOCOL_PARSERS_H
```

### Analyzer Engine (include/analyzer_engine.h)

```c
#ifndef ANALYZER_ENGINE_H
#define ANALYZER_ENGINE_H

#include "common.h"
#include "protocols.h"

// Analysis rules
typedef enum {
    ANALYSIS_RULE_PORT_SCAN = 0,
    ANALYSIS_RULE_DOS_ATTACK,
    ANALYSIS_RULE_SUSPICIOUS_PAYLOAD,
    ANALYSIS_RULE_UNUSUAL_TRAFFIC,
    ANALYSIS_RULE_MALFORMED_PACKET,
    ANALYSIS_RULE_PROTOCOL_VIOLATION,
    ANALYSIS_RULE_MAX
} analysis_rule_t;

// Analysis results
typedef struct analysis_result {
    analysis_rule_t rule;
    int severity;  // 1-10 scale
    char description[256];
    char recommendation[256];
    struct timeval timestamp;
    parsed_packet_t* related_packet;
    struct analysis_result* next;
} analysis_result_t;

// Connection tracking
typedef struct connection_info {
    uint32_t src_ip;
    uint32_t dest_ip;
    uint16_t src_port;
    uint16_t dest_port;
    uint8_t protocol;
    
    // Connection state
    uint32_t packets_sent;
    uint32_t packets_received;
    uint64_t bytes_sent;
    uint64_t bytes_received;
    
    // TCP-specific
    uint32_t seq_num;
    uint32_t ack_num;
    uint8_t tcp_state;
    
    // Timing
    struct timeval first_seen;
    struct timeval last_seen;
    
    // Flags
    int is_suspicious;
    
    // Hash table linkage
    struct connection_info* next;
} connection_info_t;

// Traffic statistics
typedef struct traffic_stats {
    // General statistics
    uint64_t total_connections;
    uint64_t active_connections;
    uint64_t completed_connections;
    uint64_t failed_connections;
    
    // Protocol distribution
    uint64_t protocol_counts[256];
    uint64_t port_counts[65536];
    
    // Top talkers
    struct {
        uint32_t ip;
        uint64_t bytes;
        uint64_t packets;
    } top_sources[10];
    
    struct {
        uint32_t ip;
        uint64_t bytes;
        uint64_t packets;
    } top_destinations[10];
    
    // Geographic distribution (if GeoIP available)
    struct {
        char country[32];
        uint64_t packets;
    } country_stats[20];
    
    pthread_mutex_t mutex;
} traffic_stats_t;

// Analyzer engine
typedef struct analyzer_engine {
    // Configuration
    int max_connections;
    int connection_timeout;
    int analysis_enabled;
    
    // Connection tracking
    connection_info_t** connection_table;
    int connection_table_size;
    pthread_mutex_t connection_mutex;
    
    // Analysis results
    analysis_result_t* results_head;
    analysis_result_t* results_tail;
    int max_results;
    pthread_mutex_t results_mutex;
    
    // Statistics
    traffic_stats_t stats;
    
    // Analysis rules
    int enabled_rules[ANALYSIS_RULE_MAX];
    int rule_thresholds[ANALYSIS_RULE_MAX];
    
    // Processing thread
    pthread_t analysis_thread;
    volatile int running;
    
    // Packet queue for analysis
    parsed_packet_t* analysis_queue_head;
    parsed_packet_t* analysis_queue_tail;
    int analysis_queue_size;
    pthread_mutex_t analysis_queue_mutex;
    pthread_cond_t analysis_queue_cond;
} analyzer_engine_t;

// Function prototypes
analyzer_engine_t* create_analyzer_engine(int max_connections);
void destroy_analyzer_engine(analyzer_engine_t* engine);
int start_analyzer_engine(analyzer_engine_t* engine);
int stop_analyzer_engine(analyzer_engine_t* engine);
void submit_packet_for_analysis(analyzer_engine_t* engine, parsed_packet_t* packet);
analysis_result_t* get_analysis_results(analyzer_engine_t* engine, int* count);
void clear_analysis_results(analyzer_engine_t* engine);
traffic_stats_t get_traffic_stats(analyzer_engine_t* engine);
void enable_analysis_rule(analyzer_engine_t* engine, analysis_rule_t rule, int threshold);
void disable_analysis_rule(analyzer_engine_t* engine, analysis_rule_t rule);

// Analysis functions
void analyze_packet(analyzer_engine_t* engine, parsed_packet_t* packet);
void detect_port_scan(analyzer_engine_t* engine, parsed_packet_t* packet);
void detect_dos_attack(analyzer_engine_t* engine, parsed_packet_t* packet);
void detect_suspicious_payload(analyzer_engine_t* engine, parsed_packet_t* packet);
void detect_malformed_packet(analyzer_engine_t* engine, parsed_packet_t* packet);
void update_connection_tracking(analyzer_engine_t* engine, parsed_packet_t* packet);
void update_traffic_statistics(analyzer_engine_t* engine, parsed_packet_t* packet);

#endif // ANALYZER_ENGINE_H
```

## Core Implementation

### Main Application (src/main.c)

```c
#include "common.h"
#include "packet_capture.h"
#include "protocol_parsers.h"
#include "analyzer_engine.h"
#include "display_manager.h"
#include "config_manager.h"
#include "log_manager.h"

volatile sig_atomic_t g_running = 1;
int g_debug_mode = 0;
int g_verbose_mode = 0;

// Global components
static packet_capture_t* g_capture = NULL;
static analyzer_engine_t* g_analyzer = NULL;
static display_manager_t* g_display = NULL;
static log_manager_t* g_logger = NULL;

void signal_handler(int sig) {
    printf("\nShutting down network analyzer...\n");
    g_running = 0;
    
    if (g_capture) {
        stop_packet_capture(g_capture);
    }
    if (g_analyzer) {
        stop_analyzer_engine(g_analyzer);
    }
    if (g_display) {
        stop_display_manager(g_display);
    }
}

void setup_signal_handlers(void) {
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);
}

void packet_handler(const uint8_t* packet, size_t length, 
                   const struct timeval* timestamp) {
    // Parse the packet
    parsed_packet_t* parsed = parse_packet(packet, length, timestamp);
    if (!parsed) {
        return;
    }
    
    // Submit to analyzer
    if (g_analyzer) {
        submit_packet_for_analysis(g_analyzer, parsed);
    }
    
    // Submit to display manager
    if (g_display) {
        submit_packet_for_display(g_display, parsed);
    }
    
    // Log if configured
    if (g_logger) {
        log_packet(g_logger, parsed);
    }
}

void error_handler(const char* error_msg) {
    fprintf(stderr, "Capture error: %s\n", error_msg);
    if (g_logger) {
        log_error(g_logger, error_msg);
    }
}

void print_usage(const char* program_name) {
    printf("Network Protocol Analyzer\n");
    printf("Usage: %s [options]\n", program_name);
    printf("\nOptions:\n");
    printf("  -i <interface>    Capture interface (default: first available)\n");
    printf("  -f <filter>       BPF filter expression\n");
    printf("  -c <config>       Configuration file\n");
    printf("  -o <output>       Output file for captured packets\n");
    printf("  -l <logfile>      Log file for analysis results\n");
    printf("  -n <count>        Number of packets to capture (0 = unlimited)\n");
    printf("  -s <size>         Snap length (max bytes per packet)\n");
    printf("  -t <timeout>      Read timeout in milliseconds\n");
    printf("  -p                Promiscuous mode\n");
    printf("  -d                Debug mode\n");
    printf("  -v                Verbose mode\n");
    printf("  -q                Quiet mode (no display)\n");
    printf("  -h                Show this help\n");
    printf("\nExamples:\n");
    printf("  %s -i eth0 -f \"tcp port 80\"\n", program_name);
    printf("  %s -i wlan0 -f \"host 192.168.1.1\" -o capture.pcap\n", program_name);
    printf("  %s -c analyzer.conf -l analysis.log\n", program_name);
}

int main(int argc, char* argv[]) {
    char interface[MAX_INTERFACE_NAME] = {0};
    char filter_expression[MAX_FILTER_LENGTH] = {0};
    char config_file[256] = "configs/analyzer.conf";
    char output_file[256] = {0};
    char log_file[256] = {0};
    int packet_count = 0;
    int snap_length = MAX_PACKET_SIZE;
    int timeout_ms = 1000;
    int promiscuous = 0;
    int quiet_mode = 0;
    
    // Parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "i:f:c:o:l:n:s:t:pdvqh")) != -1) {
        switch (opt) {
            case 'i':
                strncpy(interface, optarg, sizeof(interface) - 1);
                break;
            case 'f':
                strncpy(filter_expression, optarg, sizeof(filter_expression) - 1);
                break;
            case 'c':
                strncpy(config_file, optarg, sizeof(config_file) - 1);
                break;
            case 'o':
                strncpy(output_file, optarg, sizeof(output_file) - 1);
                break;
            case 'l':
                strncpy(log_file, optarg, sizeof(log_file) - 1);
                break;
            case 'n':
                packet_count = atoi(optarg);
                break;
            case 's':
                snap_length = atoi(optarg);
                break;
            case 't':
                timeout_ms = atoi(optarg);
                break;
            case 'p':
                promiscuous = 1;
                break;
            case 'd':
                g_debug_mode = 1;
                break;
            case 'v':
                g_verbose_mode = 1;
                break;
            case 'q':
                quiet_mode = 1;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    // Check for root privileges (required for raw sockets)
    if (geteuid() != 0) {
        fprintf(stderr, "Error: This program requires root privileges to capture packets.\n");
        fprintf(stderr, "Please run as root or with sudo.\n");
        return 1;
    }
    
    setup_signal_handlers();
    
    // Load configuration
    config_t* config = load_config(config_file);
    if (!config) {
        fprintf(stderr, "Warning: Could not load configuration file, using defaults\n");
        config = create_default_config();
    }
    
    // Apply command line overrides to config
    if (strlen(interface) > 0) {
        strncpy(config->interface, interface, sizeof(config->interface) - 1);
    }
    if (strlen(filter_expression) > 0) {
        strncpy(config->filter_expression, filter_expression, 
                sizeof(config->filter_expression) - 1);
    }
    
    // Select interface if not specified
    if (strlen(config->interface) == 0) {
        char interfaces[10][MAX_INTERFACE_NAME];
        int interface_count = list_network_interfaces(interfaces, 10);
        if (interface_count > 0) {
            strncpy(config->interface, interfaces[0], sizeof(config->interface) - 1);
            printf("Using interface: %s\n", config->interface);
        } else {
            fprintf(stderr, "Error: No network interfaces found\n");
            return 1;
        }
    }
    
    // Validate interface
    if (!validate_interface(config->interface)) {
        fprintf(stderr, "Error: Invalid interface: %s\n", config->interface);
        return 1;
    }
    
    // Initialize logger
    if (strlen(log_file) > 0 || config->enable_logging) {
        log_config_t log_config = {0};
        strncpy(log_config.log_file, strlen(log_file) > 0 ? log_file : config->log_file,
                sizeof(log_config.log_file) - 1);
        log_config.log_level = config->log_level;
        log_config.enable_packet_logging = config->enable_packet_logging;
        
        g_logger = create_log_manager(&log_config);
        if (!g_logger) {
            fprintf(stderr, "Warning: Could not initialize logging\n");
        }
    }
    
    // Initialize analyzer engine
    g_analyzer = create_analyzer_engine(config->max_connections);
    if (!g_analyzer) {
        fprintf(stderr, "Error: Could not initialize analyzer engine\n");
        return 1;
    }
    
    // Configure analysis rules
    for (int i = 0; i < ANALYSIS_RULE_MAX; i++) {
        if (config->analysis_rules[i].enabled) {
            enable_analysis_rule(g_analyzer, i, config->analysis_rules[i].threshold);
        }
    }
    
    // Initialize display manager (unless quiet mode)
    if (!quiet_mode) {
        display_config_t display_config = {0};
        display_config.refresh_interval = config->display_refresh_interval;
        display_config.max_displayed_packets = config->max_displayed_packets;
        display_config.enable_colors = config->enable_colors;
        
        g_display = create_display_manager(&display_config);
        if (!g_display) {
            fprintf(stderr, "Warning: Could not initialize display manager\n");
        }
    }
    
    // Initialize packet capture
    capture_config_t capture_config = {0};
    strncpy(capture_config.interface, config->interface, 
            sizeof(capture_config.interface) - 1);
    strncpy(capture_config.filter_expression, config->filter_expression,
            sizeof(capture_config.filter_expression) - 1);
    capture_config.promiscuous_mode = promiscuous;
    capture_config.buffer_size = config->buffer_size;
    capture_config.timeout_ms = timeout_ms;
    capture_config.max_packet_size = snap_length;
    capture_config.capture_limit = packet_count;
    capture_config.packet_handler = packet_handler;
    capture_config.error_handler = error_handler;
    
    g_capture = create_packet_capture(&capture_config);
    if (!g_capture) {
        fprintf(stderr, "Error: Could not initialize packet capture\n");
        goto cleanup;
    }
    
    // Start components
    printf("Starting Network Protocol Analyzer...\n");
    printf("Interface: %s\n", config->interface);
    printf("Filter: %s\n", strlen(config->filter_expression) > 0 ? 
           config->filter_expression : "none");
    printf("Promiscuous mode: %s\n", promiscuous ? "enabled" : "disabled");
    printf("Analysis: %s\n", config->enable_analysis ? "enabled" : "disabled");
    printf("\nPress Ctrl+C to stop capture...\n\n");
    
    if (start_analyzer_engine(g_analyzer) != 0) {
        fprintf(stderr, "Error: Could not start analyzer engine\n");
        goto cleanup;
    }
    
    if (g_display && start_display_manager(g_display) != 0) {
        fprintf(stderr, "Warning: Could not start display manager\n");
    }
    
    if (start_packet_capture(g_capture) != 0) {
        fprintf(stderr, "Error: Could not start packet capture\n");
        goto cleanup;
    }
    
    // Main loop
    time_t last_stats_print = time(NULL);
    while (g_running) {
        sleep(1);
        
        // Print statistics periodically
        if (!quiet_mode && time(NULL) - last_stats_print >= STATS_UPDATE_INTERVAL) {
            capture_stats_t stats = get_capture_stats(g_capture);
            printf("\rPackets: %lu, Bytes: %lu, PPS: %lu, BPS: %lu, Dropped: %lu",
                   stats.total_packets, stats.total_bytes, stats.packets_per_second,
                   stats.bytes_per_second, stats.dropped_packets);
            fflush(stdout);
            last_stats_print = time(NULL);
        }
    }
    
    printf("\nCapture stopped.\n");
    
    // Print final statistics
    capture_stats_t final_stats = get_capture_stats(g_capture);
    printf("\nFinal Statistics:\n");
    printf("Total packets captured: %lu\n", final_stats.total_packets);
    printf("Total bytes captured: %lu\n", final_stats.total_bytes);
    printf("Packets dropped: %lu\n", final_stats.dropped_packets);
    printf("Packet errors: %lu\n", final_stats.error_packets);
    
    // Print analysis results
    if (g_analyzer) {
        int result_count;
        analysis_result_t* results = get_analysis_results(g_analyzer, &result_count);
        if (result_count > 0) {
            printf("\nAnalysis Results (%d issues found):\n", result_count);
            analysis_result_t* current = results;
            while (current) {
                printf("  [%s] Severity: %d - %s\n", 
                       ctime(&current->timestamp.tv_sec), 
                       current->severity, current->description);
                current = current->next;
            }
        }
    }
    
cleanup:
    // Cleanup
    if (g_capture) {
        stop_packet_capture(g_capture);
        destroy_packet_capture(g_capture);
    }
    
    if (g_analyzer) {
        stop_analyzer_engine(g_analyzer);
        destroy_analyzer_engine(g_analyzer);
    }
    
    if (g_display) {
        stop_display_manager(g_display);
        destroy_display_manager(g_display);
    }
    
    if (g_logger) {
        destroy_log_manager(g_logger);
    }
    
    if (config) {
        free_config(config);
    }
    
    return 0;
}
```

### Protocol Parser Implementation (src/protocol_parsers.c)

```c
#include "protocol_parsers.h"

parsed_packet_t* parse_packet(const uint8_t* packet, size_t length, 
                             const struct timeval* timestamp) {
    if (!packet || length == 0) {
        return NULL;
    }
    
    parsed_packet_t* parsed = calloc(1, sizeof(parsed_packet_t));
    if (!parsed) {
        return NULL;
    }
    
    parsed->packet_length = length;
    parsed->timestamp = *timestamp;
    
    // Start with Ethernet frame parsing
    if (parse_ethernet_frame(packet, length, parsed) != 0) {
        free_parsed_packet(parsed);
        return NULL;
    }
    
    return parsed;
}

int parse_ethernet_frame(const uint8_t* packet, size_t length, 
                        parsed_packet_t* parsed) {
    if (length < sizeof(ethernet_header_t)) {
        return -1;
    }
    
    parsed->eth_header = (ethernet_header_t*)packet;
    parsed->packet_type = PACKET_TYPE_ETHERNET;
    
    uint16_t ethertype = ntohs(parsed->eth_header->ethertype);
    
    // Parse next layer based on EtherType
    const uint8_t* next_layer = packet + sizeof(ethernet_header_t);
    size_t next_length = length - sizeof(ethernet_header_t);
    
    switch (ethertype) {
        case PROTOCOL_IP:
            return parse_ip_packet(next_layer, next_length, parsed);
        case PROTOCOL_ARP:
            return parse_arp_packet(next_layer, next_length, parsed);
        case PROTOCOL_IPV6:
            // IPv6 parsing not implemented in this example
            return 0;
        default:
            return 0; // Unknown protocol
    }
}

int parse_ip_packet(const uint8_t* packet, size_t length, 
                   parsed_packet_t* parsed) {
    if (length < sizeof(ip_header_t)) {
        return -1;
    }
    
    parsed->ip_header = (ip_header_t*)packet;
    parsed->packet_type = PACKET_TYPE_IP;
    
    // Extract IP addresses
    struct in_addr src_addr = {.s_addr = parsed->ip_header->src_ip};
    struct in_addr dest_addr = {.s_addr = parsed->ip_header->dest_ip};
    
    inet_ntop(AF_INET, &src_addr, parsed->src_ip_str, INET_ADDRSTRLEN);
    inet_ntop(AF_INET, &dest_addr, parsed->dest_ip_str, INET_ADDRSTRLEN);
    
    // Calculate IP header length
    int ip_header_length = (parsed->ip_header->version_ihl & 0x0F) * 4;
    if (ip_header_length < 20 || ip_header_length > length) {
        return -1; // Invalid header length
    }
    
    // Verify checksum
    if (calculate_ip_checksum(parsed->ip_header) != 0) {
        parsed->is_suspicious = 1;
        parsed->analysis_notes = strdup("IP checksum error");
    }
    
    // Parse next layer based on protocol
    const uint8_t* next_layer = packet + ip_header_length;
    size_t next_length = length - ip_header_length;
    
    switch (parsed->ip_header->protocol) {
        case IPPROTO_TCP:
            return parse_tcp_segment(next_layer, next_length, parsed);
        case IPPROTO_UDP:
            return parse_udp_datagram(next_layer, next_length, parsed);
        case IPPROTO_ICMP:
            return parse_icmp_packet(next_layer, next_length, parsed);
        default:
            return 0;
    }
}

int parse_tcp_segment(const uint8_t* packet, size_t length, 
                     parsed_packet_t* parsed) {
    if (length < sizeof(tcp_header_t)) {
        return -1;
    }
    
    parsed->tcp_header = (tcp_header_t*)packet;
    parsed->packet_type = PACKET_TYPE_TCP;
    
    parsed->src_port = ntohs(parsed->tcp_header->src_port);
    parsed->dest_port = ntohs(parsed->tcp_header->dest_port);
    
    // Calculate TCP header length
    int tcp_header_length = ((parsed->tcp_header->data_offset_flags >> 4) & 0x0F) * 4;
    if (tcp_header_length < 20 || tcp_header_length > length) {
        return -1;
    }
    
    // Extract payload
    if (length > tcp_header_length) {
        parsed->payload = (char*)(packet + tcp_header_length);
        parsed->payload_length = length - tcp_header_length;
        
        // Identify application protocol
        if (parsed->src_port == 80 || parsed->dest_port == 80 ||
            parsed->src_port == 8080 || parsed->dest_port == 8080) {
            if (is_http_request(parsed->payload, parsed->payload_length) ||
                is_http_response(parsed->payload, parsed->payload_length)) {
                parse_http_message(parsed->payload, parsed->payload_length, parsed);
            }
        }
    }
    
    return 0;
}

int parse_udp_datagram(const uint8_t* packet, size_t length, 
                      parsed_packet_t* parsed) {
    if (length < sizeof(udp_header_t)) {
        return -1;
    }
    
    parsed->udp_header = (udp_header_t*)packet;
    parsed->packet_type = PACKET_TYPE_UDP;
    
    parsed->src_port = ntohs(parsed->udp_header->src_port);
    parsed->dest_port = ntohs(parsed->udp_header->dest_port);
    
    // Extract payload
    if (length > sizeof(udp_header_t)) {
        parsed->payload = (char*)(packet + sizeof(udp_header_t));
        parsed->payload_length = length - sizeof(udp_header_t);
        
        // Identify application protocol
        if (parsed->src_port == 53 || parsed->dest_port == 53) {
            parse_dns_message(parsed->payload, parsed->payload_length, parsed);
        } else if (parsed->src_port == 67 || parsed->dest_port == 67 ||
                   parsed->src_port == 68 || parsed->dest_port == 68) {
            parse_dhcp_message(parsed->payload, parsed->payload_length, parsed);
        }
    }
    
    return 0;
}

int parse_http_message(const uint8_t* payload, size_t length, 
                      parsed_packet_t* parsed) {
    if (!payload || length == 0) {
        return -1;
    }
    
    parsed->packet_type = PACKET_TYPE_HTTP;
    
    // Convert to string for parsing
    char* http_data = malloc(length + 1);
    if (!http_data) {
        return -1;
    }
    
    memcpy(http_data, payload, length);
    http_data[length] = '\0';
    
    // Check if it's a request or response
    if (is_http_request(payload, length)) {
        // Parse HTTP request
        char* method = strtok(http_data, " ");
        if (method) {
            parsed->app_data.http.method = get_http_method(method);
            
            char* url = strtok(NULL, " ");
            if (url) {
                parsed->app_data.http.url = strdup(url);
            }
        }
        
        // Extract User-Agent
        char* user_agent = extract_http_header(http_data, "User-Agent");
        if (user_agent) {
            parsed->app_data.http.user_agent = strdup(user_agent);
            free(user_agent);
        }
    } else if (is_http_response(payload, length)) {
        // Parse HTTP response
        char* status_line = strtok(http_data, "\r\n");
        if (status_line) {
            char* status_code_str = strtok(status_line, " ");
            status_code_str = strtok(NULL, " "); // Skip HTTP version
            if (status_code_str) {
                parsed->app_data.http.status_code = atoi(status_code_str);
            }
        }
    }
    
    free(http_data);
    return 0;
}

// Utility functions
int is_http_request(const uint8_t* payload, size_t length) {
    if (length < 4) return 0;
    
    const char* methods[] = {"GET ", "POST ", "PUT ", "DELETE ", "HEAD ", 
                            "OPTIONS ", "PATCH ", "TRACE ", "CONNECT "};
    
    for (int i = 0; i < sizeof(methods) / sizeof(methods[0]); i++) {
        size_t method_len = strlen(methods[i]);
        if (length >= method_len && 
            memcmp(payload, methods[i], method_len) == 0) {
            return 1;
        }
    }
    
    return 0;
}

int is_http_response(const uint8_t* payload, size_t length) {
    if (length < 8) return 0;
    
    return (memcmp(payload, "HTTP/", 5) == 0);
}

http_method_t get_http_method(const char* method_str) {
    if (strcmp(method_str, "GET") == 0) return HTTP_METHOD_GET;
    if (strcmp(method_str, "POST") == 0) return HTTP_METHOD_POST;
    if (strcmp(method_str, "PUT") == 0) return HTTP_METHOD_PUT;
    if (strcmp(method_str, "DELETE") == 0) return HTTP_METHOD_DELETE;
    if (strcmp(method_str, "HEAD") == 0) return HTTP_METHOD_HEAD;
    if (strcmp(method_str, "OPTIONS") == 0) return HTTP_METHOD_OPTIONS;
    if (strcmp(method_str, "PATCH") == 0) return HTTP_METHOD_PATCH;
    
    return HTTP_METHOD_UNKNOWN;
}

uint16_t calculate_ip_checksum(const ip_header_t* ip_header) {
    uint32_t sum = 0;
    uint16_t* ptr = (uint16_t*)ip_header;
    int length = (ip_header->version_ihl & 0x0F) * 4;
    
    // Save original checksum and zero it
    uint16_t original_checksum = ip_header->checksum;
    ((ip_header_t*)ip_header)->checksum = 0;
    
    // Calculate checksum
    for (int i = 0; i < length / 2; i++) {
        sum += ntohs(ptr[i]);
    }
    
    // Handle odd length
    if (length % 2) {
        sum += ((uint8_t*)ip_header)[length - 1] << 8;
    }
    
    // Fold carries
    while (sum >> 16) {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    
    // Restore original checksum
    ((ip_header_t*)ip_header)->checksum = original_checksum;
    
    return (uint16_t)(~sum);
}

void free_parsed_packet(parsed_packet_t* packet) {
    if (!packet) return;
    
    if (packet->app_data.http.url) {
        free(packet->app_data.http.url);
    }
    if (packet->app_data.http.user_agent) {
        free(packet->app_data.http.user_agent);
    }
    if (packet->analysis_notes) {
        free(packet->analysis_notes);
    }
    
    free(packet);
}
```

### Enhanced Makefile

```makefile
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -pthread -g -D_GNU_SOURCE
LDFLAGS = -pthread -lpcap -lncurses -lm

SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin
SCRIPTDIR = scripts

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

# Main target
TARGET = $(BINDIR)/network_analyzer

.PHONY: all clean install setup test

all: directories $(TARGET)

directories:
	@mkdir -p $(OBJDIR) $(BINDIR) configs filters

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

install: $(TARGET)
	cp $(TARGET) /usr/local/bin/
	cp -r configs /etc/network_analyzer/
	cp -r filters /etc/network_analyzer/
	chmod +x $(SCRIPTDIR)/setup_permissions.sh
	$(SCRIPTDIR)/setup_permissions.sh

setup:
	@echo "Setting up network analyzer..."
	@mkdir -p configs filters
	@chmod +x $(SCRIPTDIR)/setup_permissions.sh
	@$(SCRIPTDIR)/setup_permissions.sh

test: $(TARGET)
	@echo "Testing network analyzer..."
	@python3 $(SCRIPTDIR)/generate_traffic.py &
	@sleep 2
	@timeout 10 sudo ./$(TARGET) -i lo -f "tcp port 8080" -q || true
	@pkill -f generate_traffic.py || true

# Create default configuration files
config:
	@echo "Creating default configuration files..."
	@mkdir -p configs filters
	@cat > configs/analyzer.conf << 'EOF'
# Network Protocol Analyzer Configuration

# Network Interface
interface = eth0

# Capture Settings
promiscuous_mode = 1
buffer_size = 2097152
max_packet_size = 65536
capture_timeout = 1000

# Analysis Settings
enable_analysis = 1
max_connections = 100000
connection_timeout = 300

# Display Settings
enable_display = 1
display_refresh_interval = 100
max_displayed_packets = 1000
enable_colors = 1

# Logging Settings
enable_logging = 1
log_file = /var/log/network_analyzer.log
log_level = 2
enable_packet_logging = 0

# Analysis Rules
analysis_rules = {
    port_scan = { enabled = 1, threshold = 10 }
    dos_attack = { enabled = 1, threshold = 1000 }
    suspicious_payload = { enabled = 1, threshold = 1 }
    malformed_packet = { enabled = 1, threshold = 1 }
}
EOF

	@cat > filters/http.filter << 'EOF'
# HTTP Traffic Filter
tcp port 80 or tcp port 8080 or tcp port 443
EOF

	@cat > filters/dns.filter << 'EOF'
# DNS Traffic Filter
udp port 53 or tcp port 53
EOF

	@cat > $(SCRIPTDIR)/setup_permissions.sh << 'EOF'
#!/bin/bash
# Setup script for network analyzer

echo "Setting up permissions for network analyzer..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root"
    exit 1
fi

# Create necessary directories
mkdir -p /var/log/network_analyzer
mkdir -p /etc/network_analyzer

# Set capabilities for packet capture
setcap cap_net_raw,cap_net_admin=eip /usr/local/bin/network_analyzer

echo "Setup completed successfully!"
EOF

	@chmod +x $(SCRIPTDIR)/setup_permissions.sh
```

## Usage Examples

### Basic Usage

```bash
# Build the analyzer
make config
make

# Capture HTTP traffic
sudo ./bin/network_analyzer -i eth0 -f "tcp port 80"

# Analyze DNS traffic with logging
sudo ./bin/network_analyzer -i eth0 -f "udp port 53" -l dns_analysis.log

# Capture packets to file
sudo ./bin/network_analyzer -i eth0 -o capture.pcap -n 1000
```

### Advanced Features

- **Real-time Protocol Analysis**: Dissects and analyzes network protocols
- **Security Analysis**: Detects potential security threats and anomalies
- **Traffic Statistics**: Provides comprehensive network traffic statistics
- **Flexible Filtering**: Supports Berkeley Packet Filter (BPF) expressions
- **Multi-threaded Processing**: Handles high-volume traffic efficiently
- **Configurable Logging**: Detailed logging and analysis reporting

This network protocol analyzer demonstrates advanced socket programming concepts and provides a comprehensive foundation for network monitoring and security analysis applications.

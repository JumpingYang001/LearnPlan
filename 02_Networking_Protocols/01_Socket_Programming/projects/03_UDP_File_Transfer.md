# UDP File Transfer Project

*Last Updated: June 21, 2025*

## Project Overview

This project implements a reliable file transfer system over UDP, demonstrating how to add reliability mechanisms to unreliable protocols. The system handles packet loss, reordering, and provides error recovery while maintaining the efficiency of UDP.

## Learning Objectives

- Understand UDP protocol characteristics and limitations
- Implement reliability mechanisms over unreliable transport
- Handle packet loss detection and retransmission
- Manage packet sequencing and reordering
- Design efficient file transfer protocols
- Practice error recovery and flow control

## Project Structure

```
udp_file_transfer/
├── src/
│   ├── file_server.c
│   ├── file_client.c
│   ├── protocol.h
│   ├── reliability.c
│   ├── reliability.h
│   └── common.h
├── test_files/
├── Makefile
└── README.md
```

## Implementation

### Protocol Definition (protocol.h)

```c
#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>
#include <sys/types.h>

#define PACKET_SIZE 1024
#define DATA_SIZE (PACKET_SIZE - sizeof(packet_header_t))
#define MAX_FILENAME_LEN 256
#define MAX_RETRIES 5
#define TIMEOUT_MS 1000
#define WINDOW_SIZE 16

// Packet types
typedef enum {
    PKT_FILE_REQUEST = 1,
    PKT_FILE_RESPONSE,
    PKT_DATA,
    PKT_ACK,
    PKT_NACK,
    PKT_FIN,
    PKT_ERROR,
    PKT_HEARTBEAT
} packet_type_t;

// Packet header structure
typedef struct {
    uint8_t type;
    uint8_t flags;
    uint16_t sequence;
    uint32_t ack_number;
    uint32_t file_size;
    uint16_t data_length;
    uint16_t checksum;
    char filename[MAX_FILENAME_LEN];
} __attribute__((packed)) packet_header_t;

// Complete packet structure
typedef struct {
    packet_header_t header;
    uint8_t data[DATA_SIZE];
} __attribute__((packed)) packet_t;

// Packet flags
#define FLAG_SYN     0x01
#define FLAG_FIN     0x02
#define FLAG_RESET   0x04
#define FLAG_MORE    0x08

// Function prototypes
uint16_t calculate_checksum(const void* data, size_t length);
int verify_checksum(const packet_t* packet);
void create_packet(packet_t* packet, packet_type_t type, uint16_t sequence,
                  const char* filename, const void* data, size_t data_len);
void print_packet_info(const packet_t* packet, const char* direction);

#endif // PROTOCOL_H
```

### Reliability Manager (reliability.h)

```c
#ifndef RELIABILITY_H
#define RELIABILITY_H

#include "protocol.h"
#include <time.h>
#include <pthread.h>

// Sliding window entry
typedef struct window_entry {
    packet_t packet;
    time_t send_time;
    int retry_count;
    int acknowledged;
    struct window_entry* next;
} window_entry_t;

// Reliability context
typedef struct {
    // Sending window
    window_entry_t* send_window[WINDOW_SIZE];
    uint16_t send_base;
    uint16_t next_seq_num;
    
    // Receiving window
    packet_t* recv_window[WINDOW_SIZE];
    uint16_t recv_base;
    
    // Statistics
    uint32_t packets_sent;
    uint32_t packets_received;
    uint32_t packets_retransmitted;
    uint32_t packets_dropped;
    uint32_t bytes_transferred;
    
    // Synchronization
    pthread_mutex_t send_mutex;
    pthread_mutex_t recv_mutex;
    
    // Configuration
    int timeout_ms;
    int max_retries;
    double loss_rate;  // Simulated packet loss for testing
} reliability_ctx_t;

// Function prototypes
reliability_ctx_t* create_reliability_context(void);
void destroy_reliability_context(reliability_ctx_t* ctx);
int reliable_send(reliability_ctx_t* ctx, int sockfd, const struct sockaddr* dest,
                 socklen_t dest_len, const packet_t* packet);
int reliable_receive(reliability_ctx_t* ctx, int sockfd, packet_t* packet,
                    struct sockaddr* src, socklen_t* src_len);
void process_ack(reliability_ctx_t* ctx, uint16_t ack_num);
void check_timeouts(reliability_ctx_t* ctx, int sockfd, const struct sockaddr* dest,
                   socklen_t dest_len);
void print_statistics(const reliability_ctx_t* ctx);

#endif // RELIABILITY_H
```

### Common Header (common.h)

```c
#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>
#include <time.h>
#include <fcntl.h>
#include <sys/time.h>

#include "protocol.h"
#include "reliability.h"

#define DEFAULT_PORT 9999
#define BUFFER_SIZE 4096
#define MAX_CONNECTIONS 10

// Utility functions
void error_exit(const char* message);
void log_message(const char* format, ...);
long get_file_size(const char* filename);
int file_exists(const char* filename);
char* get_timestamp(void);
void simulate_packet_loss(double loss_rate);

// Global variables
extern volatile sig_atomic_t running;

#endif // COMMON_H
```

### Reliability Implementation (reliability.c)

```c
#include "reliability.h"
#include "common.h"

reliability_ctx_t* create_reliability_context(void) {
    reliability_ctx_t* ctx = malloc(sizeof(reliability_ctx_t));
    if (!ctx) return NULL;
    
    memset(ctx, 0, sizeof(reliability_ctx_t));
    
    ctx->send_base = 0;
    ctx->next_seq_num = 0;
    ctx->recv_base = 0;
    ctx->timeout_ms = TIMEOUT_MS;
    ctx->max_retries = MAX_RETRIES;
    ctx->loss_rate = 0.0;  // No simulated loss by default
    
    if (pthread_mutex_init(&ctx->send_mutex, NULL) != 0) {
        free(ctx);
        return NULL;
    }
    
    if (pthread_mutex_init(&ctx->recv_mutex, NULL) != 0) {
        pthread_mutex_destroy(&ctx->send_mutex);
        free(ctx);
        return NULL;
    }
    
    return ctx;
}

void destroy_reliability_context(reliability_ctx_t* ctx) {
    if (!ctx) return;
    
    pthread_mutex_lock(&ctx->send_mutex);
    
    // Clean up send window
    for (int i = 0; i < WINDOW_SIZE; i++) {
        window_entry_t* entry = ctx->send_window[i];
        while (entry) {
            window_entry_t* next = entry->next;
            free(entry);
            entry = next;
        }
    }
    
    pthread_mutex_unlock(&ctx->send_mutex);
    
    pthread_mutex_lock(&ctx->recv_mutex);
    
    // Clean up receive window
    for (int i = 0; i < WINDOW_SIZE; i++) {
        if (ctx->recv_window[i]) {
            free(ctx->recv_window[i]);
        }
    }
    
    pthread_mutex_unlock(&ctx->recv_mutex);
    
    pthread_mutex_destroy(&ctx->send_mutex);
    pthread_mutex_destroy(&ctx->recv_mutex);
    free(ctx);
}

int reliable_send(reliability_ctx_t* ctx, int sockfd, const struct sockaddr* dest,
                 socklen_t dest_len, const packet_t* packet) {
    if (!ctx || !packet) return -1;
    
    // Simulate packet loss for testing
    if (ctx->loss_rate > 0.0) {
        double random = (double)rand() / RAND_MAX;
        if (random < ctx->loss_rate) {
            log_message("Simulating packet loss (seq: %d)", packet->header.sequence);
            ctx->packets_dropped++;
            return 0;  // Pretend we sent it
        }
    }
    
    pthread_mutex_lock(&ctx->send_mutex);
    
    uint16_t seq = packet->header.sequence;
    int window_index = seq % WINDOW_SIZE;
    
    // Create window entry
    window_entry_t* entry = malloc(sizeof(window_entry_t));
    if (!entry) {
        pthread_mutex_unlock(&ctx->send_mutex);
        return -1;
    }
    
    entry->packet = *packet;
    entry->send_time = time(NULL);
    entry->retry_count = 0;
    entry->acknowledged = 0;
    entry->next = ctx->send_window[window_index];
    ctx->send_window[window_index] = entry;
    
    pthread_mutex_unlock(&ctx->send_mutex);
    
    ssize_t sent = sendto(sockfd, packet, sizeof(packet_t), 0, dest, dest_len);
    if (sent < 0) {
        log_message("sendto failed: %s", strerror(errno));
        return -1;
    }
    
    ctx->packets_sent++;
    ctx->bytes_transferred += packet->header.data_length;
    
    log_message("Sent packet: seq=%d, type=%d, size=%d", 
                packet->header.sequence, packet->header.type, 
                packet->header.data_length);
    
    return 0;
}

int reliable_receive(reliability_ctx_t* ctx, int sockfd, packet_t* packet,
                    struct sockaddr* src, socklen_t* src_len) {
    if (!ctx || !packet) return -1;
    
    ssize_t received = recvfrom(sockfd, packet, sizeof(packet_t), 0, src, src_len);
    if (received < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return 0;  // No data available
        }
        return -1;
    }
    
    if (received < sizeof(packet_header_t)) {
        log_message("Received incomplete packet (%zd bytes)", received);
        return -1;
    }
    
    // Verify checksum
    if (!verify_checksum(packet)) {
        log_message("Checksum verification failed for packet seq=%d", 
                   packet->header.sequence);
        ctx->packets_dropped++;
        return -1;
    }
    
    ctx->packets_received++;
    
    log_message("Received packet: seq=%d, type=%d, size=%d", 
                packet->header.sequence, packet->header.type, 
                packet->header.data_length);
    
    return received;
}

void process_ack(reliability_ctx_t* ctx, uint16_t ack_num) {
    if (!ctx) return;
    
    pthread_mutex_lock(&ctx->send_mutex);
    
    // Mark packets as acknowledged up to ack_num
    for (uint16_t seq = ctx->send_base; seq != ack_num; seq++) {
        int window_index = seq % WINDOW_SIZE;
        window_entry_t** entry_ptr = &ctx->send_window[window_index];
        
        while (*entry_ptr) {
            if ((*entry_ptr)->packet.header.sequence == seq) {
                window_entry_t* to_remove = *entry_ptr;
                *entry_ptr = (*entry_ptr)->next;
                free(to_remove);
                break;
            }
            entry_ptr = &(*entry_ptr)->next;
        }
    }
    
    ctx->send_base = ack_num;
    
    pthread_mutex_unlock(&ctx->send_mutex);
    
    log_message("Processed ACK: ack_num=%d, new send_base=%d", ack_num, ctx->send_base);
}

void check_timeouts(reliability_ctx_t* ctx, int sockfd, const struct sockaddr* dest,
                   socklen_t dest_len) {
    if (!ctx) return;
    
    time_t now = time(NULL);
    
    pthread_mutex_lock(&ctx->send_mutex);
    
    for (int i = 0; i < WINDOW_SIZE; i++) {
        window_entry_t* entry = ctx->send_window[i];
        
        while (entry) {
            if (!entry->acknowledged && 
                (now - entry->send_time) > (ctx->timeout_ms / 1000)) {
                
                if (entry->retry_count < ctx->max_retries) {
                    // Retransmit packet
                    ssize_t sent = sendto(sockfd, &entry->packet, sizeof(packet_t), 
                                        0, dest, dest_len);
                    if (sent > 0) {
                        entry->send_time = now;
                        entry->retry_count++;
                        ctx->packets_retransmitted++;
                        
                        log_message("Retransmitted packet: seq=%d, retry=%d", 
                                   entry->packet.header.sequence, entry->retry_count);
                    }
                } else {
                    log_message("Max retries exceeded for packet seq=%d", 
                               entry->packet.header.sequence);
                }
            }
            entry = entry->next;
        }
    }
    
    pthread_mutex_unlock(&ctx->send_mutex);
}

void print_statistics(const reliability_ctx_t* ctx) {
    if (!ctx) return;
    
    printf("\n=== Transfer Statistics ===\n");
    printf("Packets sent: %u\n", ctx->packets_sent);
    printf("Packets received: %u\n", ctx->packets_received);
    printf("Packets retransmitted: %u\n", ctx->packets_retransmitted);
    printf("Packets dropped: %u\n", ctx->packets_dropped);
    printf("Bytes transferred: %u\n", ctx->bytes_transferred);
    
    if (ctx->packets_sent > 0) {
        double loss_rate = (double)ctx->packets_dropped / ctx->packets_sent * 100.0;
        double retrans_rate = (double)ctx->packets_retransmitted / ctx->packets_sent * 100.0;
        printf("Packet loss rate: %.2f%%\n", loss_rate);
        printf("Retransmission rate: %.2f%%\n", retrans_rate);
    }
    printf("===========================\n\n");
}
```

### Protocol Implementation (add to protocol.h includes)

```c
#include "protocol.h"

uint16_t calculate_checksum(const void* data, size_t length) {
    const uint16_t* ptr = (const uint16_t*)data;
    uint32_t sum = 0;
    
    // Sum all 16-bit words
    while (length > 1) {
        sum += *ptr++;
        length -= 2;
    }
    
    // Add leftover byte, if any
    if (length > 0) {
        sum += *(const uint8_t*)ptr;
    }
    
    // Fold 32-bit sum to 16 bits
    while (sum >> 16) {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    
    return ~sum;
}

int verify_checksum(const packet_t* packet) {
    if (!packet) return 0;
    
    // Calculate checksum with header checksum field set to 0
    packet_t temp = *packet;
    temp.header.checksum = 0;
    
    uint16_t calculated = calculate_checksum(&temp, sizeof(packet_header_t) + 
                                           packet->header.data_length);
    
    return calculated == packet->header.checksum;
}

void create_packet(packet_t* packet, packet_type_t type, uint16_t sequence,
                  const char* filename, const void* data, size_t data_len) {
    if (!packet) return;
    
    memset(packet, 0, sizeof(packet_t));
    
    packet->header.type = type;
    packet->header.sequence = sequence;
    packet->header.data_length = data_len;
    
    if (filename) {
        strncpy(packet->header.filename, filename, MAX_FILENAME_LEN - 1);
    }
    
    if (data && data_len > 0 && data_len <= DATA_SIZE) {
        memcpy(packet->data, data, data_len);
    }
    
    // Calculate checksum (with checksum field set to 0)
    packet->header.checksum = 0;
    packet->header.checksum = calculate_checksum(packet, sizeof(packet_header_t) + data_len);
}

void print_packet_info(const packet_t* packet, const char* direction) {
    if (!packet || !direction) return;
    
    const char* type_names[] = {
        "UNKNOWN", "FILE_REQUEST", "FILE_RESPONSE", "DATA", 
        "ACK", "NACK", "FIN", "ERROR", "HEARTBEAT"
    };
    
    const char* type_name = (packet->header.type < sizeof(type_names)/sizeof(type_names[0])) 
                           ? type_names[packet->header.type] : "UNKNOWN";
    
    printf("[%s] %s: seq=%d, type=%s, len=%d, file=%s\n",
           get_timestamp(), direction, packet->header.sequence, 
           type_name, packet->header.data_length, packet->header.filename);
}
```

### File Server Implementation (file_server.c)

```c
#include "common.h"

volatile sig_atomic_t running = 1;
int server_socket = -1;

typedef struct {
    struct sockaddr_in client_addr;
    reliability_ctx_t* ctx;
    char filename[MAX_FILENAME_LEN];
    FILE* file;
    long file_size;
    long bytes_sent;
    uint16_t next_seq;
    time_t start_time;
} transfer_session_t;

void signal_handler(int signal) {
    printf("\nShutting down file server...\n");
    running = 0;
    
    if (server_socket != -1) {
        close(server_socket);
    }
}

void setup_signal_handlers(void) {
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);
}

int create_server_socket(int port) {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd == -1) {
        error_exit("socket creation failed");
    }
    
    int opt = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) == -1) {
        close(sockfd);
        error_exit("setsockopt failed");
    }
    
    // Set socket to non-blocking for timeout handling
    int flags = fcntl(sockfd, F_GETFL, 0);
    if (fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) == -1) {
        close(sockfd);
        error_exit("fcntl failed");
    }
    
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);
    
    if (bind(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
        close(sockfd);
        error_exit("bind failed");
    }
    
    return sockfd;
}

void send_file_response(int sockfd, const struct sockaddr_in* client_addr, 
                       const char* filename, int success, long file_size) {
    packet_t response;
    
    if (success) {
        create_packet(&response, PKT_FILE_RESPONSE, 0, filename, NULL, 0);
        response.header.file_size = file_size;
        response.header.flags = FLAG_SYN;
    } else {
        create_packet(&response, PKT_ERROR, 0, filename, "File not found", 14);
    }
    
    sendto(sockfd, &response, sizeof(packet_t), 0, 
           (struct sockaddr*)client_addr, sizeof(*client_addr));
    
    log_message("Sent file response for '%s': %s (size: %ld)", 
                filename, success ? "SUCCESS" : "FAILED", file_size);
}

void send_file_data(int sockfd, transfer_session_t* session) {
    if (!session || !session->file) return;
    
    char buffer[DATA_SIZE];
    size_t bytes_read = fread(buffer, 1, DATA_SIZE, session->file);
    
    if (bytes_read > 0) {
        packet_t data_packet;
        create_packet(&data_packet, PKT_DATA, session->next_seq, 
                     session->filename, buffer, bytes_read);
        
        // Set MORE flag if this is not the last packet
        if (session->bytes_sent + bytes_read < session->file_size) {
            data_packet.header.flags |= FLAG_MORE;
        } else {
            data_packet.header.flags |= FLAG_FIN;
        }
        
        if (reliable_send(session->ctx, sockfd, (struct sockaddr*)&session->client_addr,
                         sizeof(session->client_addr), &data_packet) == 0) {
            session->bytes_sent += bytes_read;
            session->next_seq++;
            
            log_message("Sent data packet: seq=%d, bytes=%zu, total=%ld/%ld", 
                       session->next_seq - 1, bytes_read, 
                       session->bytes_sent, session->file_size);
        }
    }
    
    if (session->bytes_sent >= session->file_size) {
        log_message("File transfer completed: %s (%ld bytes)", 
                   session->filename, session->file_size);
        
        double duration = difftime(time(NULL), session->start_time);
        double throughput = duration > 0 ? (session->file_size / duration) / 1024.0 : 0.0;
        log_message("Transfer time: %.2f seconds, throughput: %.2f KB/s", 
                   duration, throughput);
        
        print_statistics(session->ctx);
    }
}

void handle_client_request(int sockfd, const packet_t* packet, 
                          const struct sockaddr_in* client_addr) {
    static transfer_session_t* active_sessions[MAX_CONNECTIONS] = {0};
    
    switch (packet->header.type) {
        case PKT_FILE_REQUEST: {
            // Find free session slot
            int session_index = -1;
            for (int i = 0; i < MAX_CONNECTIONS; i++) {
                if (!active_sessions[i]) {
                    session_index = i;
                    break;
                }
            }
            
            if (session_index == -1) {
                log_message("No free session slots available");
                send_file_response(sockfd, client_addr, packet->header.filename, 0, 0);
                return;
            }
            
            // Check if file exists
            if (!file_exists(packet->header.filename)) {
                log_message("File not found: %s", packet->header.filename);
                send_file_response(sockfd, client_addr, packet->header.filename, 0, 0);
                return;
            }
            
            long file_size = get_file_size(packet->header.filename);
            if (file_size < 0) {
                log_message("Cannot get file size: %s", packet->header.filename);
                send_file_response(sockfd, client_addr, packet->header.filename, 0, 0);
                return;
            }
            
            // Create new transfer session
            transfer_session_t* session = malloc(sizeof(transfer_session_t));
            if (!session) {
                log_message("Memory allocation failed for session");
                send_file_response(sockfd, client_addr, packet->header.filename, 0, 0);
                return;
            }
            
            session->client_addr = *client_addr;
            session->ctx = create_reliability_context();
            if (!session->ctx) {
                free(session);
                send_file_response(sockfd, client_addr, packet->header.filename, 0, 0);
                return;
            }
            
            // Configure simulated packet loss for testing
            session->ctx->loss_rate = 0.05;  // 5% packet loss
            
            strncpy(session->filename, packet->header.filename, MAX_FILENAME_LEN - 1);
            session->file = fopen(packet->header.filename, "rb");
            if (!session->file) {
                destroy_reliability_context(session->ctx);
                free(session);
                send_file_response(sockfd, client_addr, packet->header.filename, 0, 0);
                return;
            }
            
            session->file_size = file_size;
            session->bytes_sent = 0;
            session->next_seq = 1;
            session->start_time = time(NULL);
            
            active_sessions[session_index] = session;
            
            log_message("Starting file transfer: %s (%ld bytes) to %s:%d", 
                       packet->header.filename, file_size,
                       inet_ntoa(client_addr->sin_addr), ntohs(client_addr->sin_port));
            
            send_file_response(sockfd, client_addr, packet->header.filename, 1, file_size);
            
            // Start sending file data
            send_file_data(sockfd, session);
            break;
        }
        
        case PKT_ACK: {
            // Find session for this client
            for (int i = 0; i < MAX_CONNECTIONS; i++) {
                transfer_session_t* session = active_sessions[i];
                if (session && 
                    session->client_addr.sin_addr.s_addr == client_addr->sin_addr.s_addr &&
                    session->client_addr.sin_port == client_addr->sin_port) {
                    
                    process_ack(session->ctx, packet->header.ack_number);
                    
                    // Send more data if available
                    if (session->bytes_sent < session->file_size) {
                        send_file_data(sockfd, session);
                    } else {
                        // Transfer completed, clean up session
                        fclose(session->file);
                        destroy_reliability_context(session->ctx);
                        free(session);
                        active_sessions[i] = NULL;
                    }
                    break;
                }
            }
            break;
        }
        
        default:
            log_message("Received unknown packet type: %d", packet->header.type);
            break;
    }
    
    // Check for timeouts in all active sessions
    for (int i = 0; i < MAX_CONNECTIONS; i++) {
        transfer_session_t* session = active_sessions[i];
        if (session) {
            check_timeouts(session->ctx, sockfd, (struct sockaddr*)&session->client_addr,
                          sizeof(session->client_addr));
        }
    }
}

int main(int argc, char* argv[]) {
    int port = DEFAULT_PORT;
    
    if (argc > 1) {
        port = atoi(argv[1]);
        if (port <= 0 || port > 65535) {
            fprintf(stderr, "Invalid port number: %s\n", argv[1]);
            exit(EXIT_FAILURE);
        }
    }
    
    setup_signal_handlers();
    
    server_socket = create_server_socket(port);
    
    log_message("UDP file server started on port %d", port);
    log_message("Serving files from current directory");
    log_message("Packet size: %d bytes, data size: %d bytes", PACKET_SIZE, DATA_SIZE);
    
    packet_t packet;
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    
    while (running) {
        int bytes_received = reliable_receive(NULL, server_socket, &packet, 
                                            (struct sockaddr*)&client_addr, &client_len);
        
        if (bytes_received > 0) {
            log_message("Received request from %s:%d", 
                       inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));
            handle_client_request(server_socket, &packet, &client_addr);
        } else if (bytes_received < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            log_message("recvfrom error: %s", strerror(errno));
        }
        
        // Small delay to prevent busy waiting
        usleep(1000);  // 1ms
    }
    
    log_message("Server shutting down...");
    
    if (server_socket != -1) {
        close(server_socket);
    }
    
    return 0;
}
```

### File Client Implementation (file_client.c)

```c
#include "common.h"

volatile sig_atomic_t running = 1;
int client_socket = -1;

typedef struct {
    char local_filename[MAX_FILENAME_LEN];
    char remote_filename[MAX_FILENAME_LEN];
    FILE* file;
    long file_size;
    long bytes_received;
    uint16_t expected_seq;
    reliability_ctx_t* ctx;
    time_t start_time;
} download_session_t;

void signal_handler(int signal) {
    printf("\nCanceling file transfer...\n");
    running = 0;
    
    if (client_socket != -1) {
        close(client_socket);
    }
}

void setup_signal_handlers(void) {
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    
    sigaction(SIGINT, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);
}

int create_client_socket(void) {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd == -1) {
        error_exit("socket creation failed");
    }
    
    // Set socket to non-blocking
    int flags = fcntl(sockfd, F_GETFL, 0);
    if (fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) == -1) {
        close(sockfd);
        error_exit("fcntl failed");
    }
    
    return sockfd;
}

void send_ack(int sockfd, const struct sockaddr_in* server_addr, uint16_t ack_num) {
    packet_t ack_packet;
    create_packet(&ack_packet, PKT_ACK, 0, NULL, NULL, 0);
    ack_packet.header.ack_number = ack_num;
    
    sendto(sockfd, &ack_packet, sizeof(packet_t), 0, 
           (struct sockaddr*)server_addr, sizeof(*server_addr));
    
    log_message("Sent ACK: ack_num=%d", ack_num);
}

void send_file_request(int sockfd, const struct sockaddr_in* server_addr, 
                      const char* filename) {
    packet_t request;
    create_packet(&request, PKT_FILE_REQUEST, 0, filename, NULL, 0);
    request.header.flags = FLAG_SYN;
    
    sendto(sockfd, &request, sizeof(packet_t), 0, 
           (struct sockaddr*)server_addr, sizeof(*server_addr));
    
    log_message("Sent file request: %s", filename);
}

void print_progress(long bytes_received, long total_size) {
    if (total_size <= 0) return;
    
    double percentage = (double)bytes_received / total_size * 100.0;
    int bar_width = 50;
    int filled = (int)(percentage / 100.0 * bar_width);
    
    printf("\rProgress: [");
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) {
            printf("=");
        } else {
            printf(" ");
        }
    }
    printf("] %.1f%% (%ld/%ld bytes)", percentage, bytes_received, total_size);
    fflush(stdout);
}

int handle_server_response(int sockfd, download_session_t* session, 
                          const struct sockaddr_in* server_addr) {
    packet_t packet;
    struct sockaddr_in src_addr;
    socklen_t src_len = sizeof(src_addr);
    
    fd_set readfds;
    struct timeval timeout;
    
    while (running) {
        FD_ZERO(&readfds);
        FD_SET(sockfd, &readfds);
        
        timeout.tv_sec = 5;  // 5 second timeout
        timeout.tv_usec = 0;
        
        int result = select(sockfd + 1, &readfds, NULL, NULL, &timeout);
        
        if (result < 0) {
            if (errno == EINTR) continue;
            log_message("select error: %s", strerror(errno));
            return -1;
        } else if (result == 0) {
            log_message("Timeout waiting for server response");
            return -1;
        }
        
        int bytes_received = reliable_receive(session->ctx, sockfd, &packet, 
                                            (struct sockaddr*)&src_addr, &src_len);
        
        if (bytes_received <= 0) continue;
        
        switch (packet.header.type) {
            case PKT_FILE_RESPONSE:
                if (packet.header.file_size > 0) {
                    session->file_size = packet.header.file_size;
                    session->expected_seq = 1;
                    session->start_time = time(NULL);
                    
                    log_message("File transfer starting: %s (%ld bytes)", 
                               session->remote_filename, session->file_size);
                    
                    // Send ACK for file response
                    send_ack(sockfd, server_addr, 0);
                    return 1;  // Success
                } else {
                    log_message("File not found on server: %s", session->remote_filename);
                    return 0;  // File not found
                }
                break;
                
            case PKT_ERROR:
                log_message("Server error: %s", packet.data);
                return -1;
                
            case PKT_DATA:
                // Handle data packet
                if (packet.header.sequence == session->expected_seq) {
                    // Write data to file
                    size_t written = fwrite(packet.data, 1, packet.header.data_length, 
                                          session->file);
                    if (written != packet.header.data_length) {
                        log_message("File write error");
                        return -1;
                    }
                    
                    session->bytes_received += packet.header.data_length;
                    session->expected_seq++;
                    
                    print_progress(session->bytes_received, session->file_size);
                    
                    // Send ACK
                    send_ack(sockfd, server_addr, session->expected_seq);
                    
                    // Check if transfer is complete
                    if (packet.header.flags & FLAG_FIN) {
                        printf("\n");
                        log_message("File transfer completed: %s", session->local_filename);
                        
                        double duration = difftime(time(NULL), session->start_time);
                        double throughput = duration > 0 ? 
                                          (session->bytes_received / duration) / 1024.0 : 0.0;
                        log_message("Transfer time: %.2f seconds, throughput: %.2f KB/s", 
                                   duration, throughput);
                        
                        print_statistics(session->ctx);
                        return 2;  // Transfer complete
                    }
                } else {
                    // Out of order packet, send ACK for last in-order packet
                    log_message("Out of order packet: expected=%d, received=%d", 
                               session->expected_seq, packet.header.sequence);
                    send_ack(sockfd, server_addr, session->expected_seq);
                }
                break;
                
            default:
                log_message("Received unknown packet type: %d", packet.header.type);
                break;
        }
        
        // Check for timeouts
        check_timeouts(session->ctx, sockfd, (struct sockaddr*)server_addr, 
                      sizeof(*server_addr));
    }
    
    return -1;
}

int download_file(const char* server_ip, int port, const char* remote_filename, 
                 const char* local_filename) {
    download_session_t session;
    memset(&session, 0, sizeof(session));
    
    strncpy(session.remote_filename, remote_filename, MAX_FILENAME_LEN - 1);
    strncpy(session.local_filename, local_filename, MAX_FILENAME_LEN - 1);
    
    // Create reliability context
    session.ctx = create_reliability_context();
    if (!session.ctx) {
        log_message("Failed to create reliability context");
        return -1;
    }
    
    // Open local file for writing
    session.file = fopen(local_filename, "wb");
    if (!session.file) {
        log_message("Cannot create local file: %s", local_filename);
        destroy_reliability_context(session.ctx);
        return -1;
    }
    
    // Create client socket
    client_socket = create_client_socket();
    
    // Setup server address
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    
    if (inet_pton(AF_INET, server_ip, &server_addr.sin_addr) <= 0) {
        log_message("Invalid server IP address: %s", server_ip);
        fclose(session.file);
        close(client_socket);
        destroy_reliability_context(session.ctx);
        return -1;
    }
    
    log_message("Connecting to server %s:%d", server_ip, port);
    log_message("Requesting file: %s -> %s", remote_filename, local_filename);
    
    // Send file request
    send_file_request(client_socket, &server_addr, remote_filename);
    
    // Handle server response and file transfer
    int result = handle_server_response(client_socket, &session, &server_addr);
    
    // Cleanup
    fclose(session.file);
    close(client_socket);
    destroy_reliability_context(session.ctx);
    
    if (result <= 0) {
        // Remove incomplete file on failure
        unlink(local_filename);
    }
    
    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 4 && argc != 5) {
        printf("Usage: %s <server_ip> <remote_filename> <local_filename> [port]\n", argv[0]);
        printf("Example: %s 192.168.1.100 test.txt downloaded_test.txt 9999\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    
    char* server_ip = argv[1];
    char* remote_filename = argv[2];
    char* local_filename = argv[3];
    int port = (argc == 5) ? atoi(argv[4]) : DEFAULT_PORT;
    
    if (port <= 0 || port > 65535) {
        fprintf(stderr, "Invalid port number: %d\n", port);
        exit(EXIT_FAILURE);
    }
    
    setup_signal_handlers();
    
    int result = download_file(server_ip, port, remote_filename, local_filename);
    
    switch (result) {
        case 2:
            printf("File download completed successfully!\n");
            break;
        case 1:
            printf("File transfer started but incomplete\n");
            break;
        case 0:
            printf("File not found on server\n");
            break;
        default:
            printf("File download failed\n");
            break;
    }
    
    return (result > 0) ? 0 : 1;
}
```

### Utility Functions (add to common.h implementation)

```c
#include "common.h"
#include <stdarg.h>

void error_exit(const char* message) {
    perror(message);
    exit(EXIT_FAILURE);
}

void log_message(const char* format, ...) {
    char* timestamp = get_timestamp();
    printf("[%s] ", timestamp);
    free(timestamp);
    
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    
    printf("\n");
    fflush(stdout);
}

long get_file_size(const char* filename) {
    struct stat st;
    if (stat(filename, &st) == 0) {
        return st.st_size;
    }
    return -1;
}

int file_exists(const char* filename) {
    return access(filename, F_OK) == 0;
}

char* get_timestamp(void) {
    time_t now = time(NULL);
    struct tm* timeinfo = localtime(&now);
    char* buffer = malloc(32);
    
    if (buffer) {
        strftime(buffer, 32, "%H:%M:%S", timeinfo);
    }
    
    return buffer;
}
```

### Enhanced Makefile

```makefile
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -pthread -g -D_GNU_SOURCE
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Source files
COMMON_SRC = $(SRCDIR)/reliability.c
SERVER_SRC = $(SRCDIR)/file_server.c $(COMMON_SRC)
CLIENT_SRC = $(SRCDIR)/file_client.c $(COMMON_SRC)

# Object files
SERVER_OBJ = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SERVER_SRC))
CLIENT_OBJ = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(CLIENT_SRC))

# Executables
SERVER_BIN = $(BINDIR)/file_server
CLIENT_BIN = $(BINDIR)/file_client

.PHONY: all clean directories test-files run-server run-client

all: directories $(SERVER_BIN) $(CLIENT_BIN)

directories:
	@mkdir -p $(OBJDIR) $(BINDIR) test_files

$(SERVER_BIN): $(SERVER_OBJ)
	$(CC) $(CFLAGS) -o $@ $^

$(CLIENT_BIN): $(CLIENT_OBJ)
	$(CC) $(CFLAGS) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJDIR) $(BINDIR)

test-files:
	@mkdir -p test_files
	@echo "Creating test files..."
	@dd if=/dev/zero of=test_files/small.txt bs=1K count=1 2>/dev/null
	@dd if=/dev/zero of=test_files/medium.txt bs=1K count=100 2>/dev/null
	@dd if=/dev/zero of=test_files/large.txt bs=1M count=10 2>/dev/null
	@echo "Test files created in test_files/"

run-server: $(SERVER_BIN)
	cd test_files && ../$(SERVER_BIN)

run-client: $(CLIENT_BIN)
	./$(CLIENT_BIN) 127.0.0.1 small.txt downloaded_small.txt
```

## Usage Instructions

### Setup and Build

```bash
# Build the project
make

# Create test files
make test-files

# Terminal 1: Start the server
make run-server

# Terminal 2: Download a file
./bin/file_client 127.0.0.1 small.txt downloaded_small.txt
```

### Features Implemented

- ✅ Reliable UDP with ACK/NACK mechanism
- ✅ Packet sequencing and reordering detection
- ✅ Timeout and retransmission handling
- ✅ Sliding window protocol
- ✅ Checksum verification for data integrity
- ✅ Progress reporting during transfer
- ✅ Transfer statistics and performance metrics
- ✅ Simulated packet loss for testing reliability
- ✅ Multiple concurrent file transfers
- ✅ Error recovery and graceful handling

This UDP file transfer project demonstrates how to build reliability on top of an unreliable protocol, including essential concepts like acknowledgments, retransmission, sequencing, and flow control.

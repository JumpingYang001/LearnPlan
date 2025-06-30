# Application Layer Protocols Overview

*Duration: 2 weeks*

## Introduction

Application layer protocols are the top layer of the TCP/IP protocol stack, providing network services directly to end-user applications. These protocols define how applications communicate over networks and are essential for modern internet functionality.

### Protocol Stack Context
```
┌─────────────────────────────────────┐
│     Application Layer               │ ← We are here
│  (HTTP, DNS, SMTP, FTP, DHCP)      │
├─────────────────────────────────────┤
│     Transport Layer (TCP/UDP)       │
├─────────────────────────────────────┤
│     Network Layer (IP)              │
├─────────────────────────────────────┤
│     Link Layer (Ethernet, WiFi)     │
└─────────────────────────────────────┘
```

## Learning Objectives

By the end of this section, you should be able to:
- **Understand the purpose and operation** of major application layer protocols
- **Implement basic protocol clients** using socket programming
- **Analyze protocol messages** and understand their structure
- **Debug network communication** at the application layer
- **Design simple application protocols** following established patterns

## Core Application Layer Protocols

## 1. Domain Name System (DNS)

### Overview
DNS translates human-readable domain names (like `www.example.com`) into IP addresses (like `192.0.2.1`) that computers use to identify each other on the network.

### DNS Message Structure

#### DNS Header Format
```c
#include <stdio.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <string.h>

// DNS Header Structure (12 bytes)
struct dns_header {
    uint16_t id;        // Transaction ID
    uint16_t flags;     // Flags (QR, Opcode, AA, TC, RD, RA, Z, RCODE)
    uint16_t qdcount;   // Number of questions
    uint16_t ancount;   // Number of answer RRs
    uint16_t nscount;   // Number of authority RRs
    uint16_t arcount;   // Number of additional RRs
} __attribute__((packed));

// DNS Question Section
struct dns_question {
    // Domain name (variable length, null-terminated)
    uint16_t qtype;     // Query type (A, AAAA, MX, etc.)
    uint16_t qclass;    // Query class (usually IN for Internet)
} __attribute__((packed));

// DNS Resource Record
struct dns_rr {
    // Name (variable length or pointer)
    uint16_t type;      // Record type
    uint16_t class;     // Record class
    uint32_t ttl;       // Time to live
    uint16_t rdlength;  // Length of RDATA
    // RDATA follows (variable length)
} __attribute__((packed));

void print_dns_header_info() {
    printf("DNS Header Structure:\n");
    printf("- Total size: %zu bytes\n", sizeof(struct dns_header));
    printf("- ID field: 16 bits (identifies request/response pair)\n");
    printf("- Flags field breakdown:\n");
    printf("  * QR (1 bit): Query(0) or Response(1)\n");
    printf("  * Opcode (4 bits): Operation code\n");
    printf("  * AA (1 bit): Authoritative Answer\n");
    printf("  * TC (1 bit): Truncated\n");
    printf("  * RD (1 bit): Recursion Desired\n");
    printf("  * RA (1 bit): Recursion Available\n");
    printf("  * Z (3 bits): Reserved\n");
    printf("  * RCODE (4 bits): Response Code\n");
}

int main() {
    print_dns_header_info();
    return 0;
}
```

### Complete DNS Query Implementation

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#define DNS_SERVER "8.8.8.8"  // Google DNS
#define DNS_PORT 53
#define MAX_DNS_SIZE 512

// DNS Query Types
#define DNS_TYPE_A 1      // IPv4 address
#define DNS_TYPE_AAAA 28  // IPv6 address
#define DNS_TYPE_MX 15    // Mail exchange
#define DNS_TYPE_CNAME 5  // Canonical name

// Convert domain name to DNS format (length-prefixed labels)
int name_to_dns_format(const char* name, char* dns_name) {
    int pos = 0;
    int label_start = 0;
    int name_len = strlen(name);
    
    for (int i = 0; i <= name_len; i++) {
        if (name[i] == '.' || name[i] == '\0') {
            int label_len = i - label_start;
            dns_name[pos++] = label_len;
            memcpy(&dns_name[pos], &name[label_start], label_len);
            pos += label_len;
            label_start = i + 1;
        }
    }
    dns_name[pos++] = 0; // Root label
    return pos;
}

// Create DNS query packet
int create_dns_query(const char* domain, uint16_t query_type, char* buffer) {
    struct dns_header* header = (struct dns_header*)buffer;
    int pos = sizeof(struct dns_header);
    
    // Fill DNS header
    header->id = htons(0x1234);        // Transaction ID
    header->flags = htons(0x0100);     // Standard query, recursion desired
    header->qdcount = htons(1);        // One question
    header->ancount = 0;
    header->nscount = 0;
    header->arcount = 0;
    
    // Add question section
    pos += name_to_dns_format(domain, &buffer[pos]);
    
    // Add QTYPE and QCLASS
    *(uint16_t*)&buffer[pos] = htons(query_type);
    pos += 2;
    *(uint16_t*)&buffer[pos] = htons(1); // IN class
    pos += 2;
    
    return pos;
}

// Parse DNS response
void parse_dns_response(char* buffer, int len) {
    struct dns_header* header = (struct dns_header*)buffer;
    
    printf("\nDNS Response Analysis:\n");
    printf("Transaction ID: 0x%04x\n", ntohs(header->id));
    printf("Flags: 0x%04x\n", ntohs(header->flags));
    printf("Questions: %d\n", ntohs(header->qdcount));
    printf("Answers: %d\n", ntohs(header->ancount));
    printf("Authority RRs: %d\n", ntohs(header->nscount));
    printf("Additional RRs: %d\n", ntohs(header->arcount));
    
    // Check response code
    uint16_t flags = ntohs(header->flags);
    int rcode = flags & 0x000F;
    
    if (rcode == 0) {
        printf("Status: Success\n");
    } else {
        printf("Status: Error (RCODE=%d)\n", rcode);
        return;
    }
    
    // Parse answers (simplified - just show IP for A records)
    if (ntohs(header->ancount) > 0) {
        printf("\nIP Addresses found:\n");
        
        int pos = sizeof(struct dns_header);
        
        // Skip question section (simplified parsing)
        while (buffer[pos] != 0) pos++; // Skip name
        pos += 5; // Skip null terminator + QTYPE + QCLASS
        
        // Parse answer section
        for (int i = 0; i < ntohs(header->ancount); i++) {
            // Skip name (could be compressed)
            if ((buffer[pos] & 0xC0) == 0xC0) {
                pos += 2; // Compressed name pointer
            } else {
                while (buffer[pos] != 0) pos++; // Regular name
                pos++;
            }
            
            uint16_t type = ntohs(*(uint16_t*)&buffer[pos]);
            pos += 2;
            uint16_t class = ntohs(*(uint16_t*)&buffer[pos]);
            pos += 2;
            uint32_t ttl = ntohl(*(uint32_t*)&buffer[pos]);
            pos += 4;
            uint16_t rdlength = ntohs(*(uint16_t*)&buffer[pos]);
            pos += 2;
            
            if (type == DNS_TYPE_A && rdlength == 4) {
                struct in_addr addr;
                memcpy(&addr, &buffer[pos], 4);
                printf("  %s (TTL: %d seconds)\n", inet_ntoa(addr), ttl);
            }
            
            pos += rdlength;
        }
    }
}

// Perform DNS lookup
int dns_lookup(const char* domain, uint16_t query_type) {
    int sockfd;
    struct sockaddr_in server_addr;
    char query_buffer[MAX_DNS_SIZE];
    char response_buffer[MAX_DNS_SIZE];
    
    // Create UDP socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        return -1;
    }
    
    // Configure DNS server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(DNS_PORT);
    inet_pton(AF_INET, DNS_SERVER, &server_addr.sin_addr);
    
    // Create DNS query
    int query_len = create_dns_query(domain, query_type, query_buffer);
    
    printf("Querying DNS for: %s\n", domain);
    printf("Query size: %d bytes\n", query_len);
    
    // Send query
    if (sendto(sockfd, query_buffer, query_len, 0, 
               (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Send failed");
        close(sockfd);
        return -1;
    }
    
    // Receive response
    socklen_t addr_len = sizeof(server_addr);
    int response_len = recvfrom(sockfd, response_buffer, MAX_DNS_SIZE, 0,
                               (struct sockaddr*)&server_addr, &addr_len);
    
    if (response_len < 0) {
        perror("Receive failed");
        close(sockfd);
        return -1;
    }
    
    printf("Response received: %d bytes\n", response_len);
    
    // Parse and display response
    parse_dns_response(response_buffer, response_len);
    
    close(sockfd);
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <domain_name>\n", argv[0]);
        printf("Example: %s www.google.com\n", argv[0]);
        return 1;
    }
    
    printf("=== DNS Lookup Tool ===\n");
    return dns_lookup(argv[1], DNS_TYPE_A);
}
```

### DNS Record Types

| Type | Name | Description | Example |
|------|------|-------------|---------|
| A | Address | IPv4 address | 192.0.2.1 |
| AAAA | IPv6 Address | IPv6 address | 2001:db8::1 |
| CNAME | Canonical Name | Alias for another name | www → example.com |
| MX | Mail Exchange | Mail server for domain | 10 mail.example.com |
| NS | Name Server | Authoritative name server | ns1.example.com |
| PTR | Pointer | Reverse DNS lookup | 1.2.0.192.in-addr.arpa |
| TXT | Text | Arbitrary text data | "v=spf1 include:_spf.google.com ~all" |

## 2. HyperText Transfer Protocol (HTTP)

### Overview
HTTP is a stateless, application-layer protocol used for distributed, collaborative, hypermedia information systems. It's the foundation of data communication on the World Wide Web.

### HTTP Message Structure

#### HTTP Request Format
```
REQUEST_LINE\r\n
HEADER_FIELD: value\r\n
HEADER_FIELD: value\r\n
\r\n
[MESSAGE_BODY]
```

#### HTTP Response Format
```
STATUS_LINE\r\n
HEADER_FIELD: value\r\n
HEADER_FIELD: value\r\n
\r\n
[MESSAGE_BODY]
```

### Complete HTTP Client Implementation

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

#define BUFFER_SIZE 4096
#define HTTP_PORT 80
#define HTTPS_PORT 443

// HTTP Request structure
typedef struct {
    char method[16];      // GET, POST, PUT, DELETE, etc.
    char path[256];       // /path/to/resource
    char host[256];       // www.example.com
    char version[16];     // HTTP/1.1
    char headers[1024];   // Additional headers
    char body[2048];      // Request body (for POST/PUT)
} http_request_t;

// HTTP Response structure
typedef struct {
    char version[16];     // HTTP/1.1
    int status_code;      // 200, 404, 500, etc.
    char status_text[64]; // OK, Not Found, etc.
    char headers[2048];   // Response headers
    char* body;           // Response body
    size_t body_length;   // Body length
} http_response_t;

// Parse URL into components
int parse_url(const char* url, char* host, char* path, int* port) {
    const char* start = url;
    
    // Skip protocol
    if (strncmp(url, "http://", 7) == 0) {
        start += 7;
        *port = HTTP_PORT;
    } else if (strncmp(url, "https://", 8) == 0) {
        start += 8;
        *port = HTTPS_PORT;
        printf("Note: HTTPS not supported in this simple implementation\n");
        return -1;
    } else {
        *port = HTTP_PORT;
    }
    
    // Find path separator
    const char* path_start = strchr(start, '/');
    if (path_start) {
        // Copy host
        int host_len = path_start - start;
        strncpy(host, start, host_len);
        host[host_len] = '\0';
        
        // Copy path
        strcpy(path, path_start);
    } else {
        // No path, use root
        strcpy(host, start);
        strcpy(path, "/");
    }
    
    // Check for port in host
    char* port_sep = strchr(host, ':');
    if (port_sep) {
        *port_sep = '\0';
        *port = atoi(port_sep + 1);
    }
    
    return 0;
}

// Create HTTP request string
int build_http_request(const http_request_t* req, char* buffer, size_t buffer_size) {
    int len = 0;
    
    // Request line
    len += snprintf(buffer + len, buffer_size - len, 
                   "%s %s %s\r\n", req->method, req->path, req->version);
    
    // Host header (required in HTTP/1.1)
    len += snprintf(buffer + len, buffer_size - len, 
                   "Host: %s\r\n", req->host);
    
    // Connection header
    len += snprintf(buffer + len, buffer_size - len, 
                   "Connection: close\r\n");
    
    // User-Agent header
    len += snprintf(buffer + len, buffer_size - len, 
                   "User-Agent: Simple-HTTP-Client/1.0\r\n");
    
    // Additional headers
    if (strlen(req->headers) > 0) {
        len += snprintf(buffer + len, buffer_size - len, "%s", req->headers);
    }
    
    // Content-Length for POST/PUT
    if (strlen(req->body) > 0) {
        len += snprintf(buffer + len, buffer_size - len, 
                       "Content-Length: %zu\r\n", strlen(req->body));
        len += snprintf(buffer + len, buffer_size - len, 
                       "Content-Type: application/x-www-form-urlencoded\r\n");
    }
    
    // Empty line to end headers
    len += snprintf(buffer + len, buffer_size - len, "\r\n");
    
    // Body (if any)
    if (strlen(req->body) > 0) {
        len += snprintf(buffer + len, buffer_size - len, "%s", req->body);
    }
    
    return len;
}

// Parse HTTP response
int parse_http_response(const char* response, http_response_t* parsed) {
    const char* line_end;
    const char* current = response;
    
    // Parse status line
    line_end = strstr(current, "\r\n");
    if (!line_end) return -1;
    
    sscanf(current, "%s %d %63[^\r\n]", 
           parsed->version, &parsed->status_code, parsed->status_text);
    
    current = line_end + 2;
    
    // Parse headers
    char* headers_start = parsed->headers;
    size_t headers_space = sizeof(parsed->headers) - 1;
    
    while ((line_end = strstr(current, "\r\n")) != NULL) {
        if (line_end == current) {
            // Empty line - end of headers
            current += 2;
            break;
        }
        
        // Copy header line
        int line_len = line_end - current;
        if (line_len < headers_space) {
            strncpy(headers_start, current, line_len);
            headers_start[line_len] = '\n';
            headers_start += line_len + 1;
            headers_space -= line_len + 1;
        }
        
        current = line_end + 2;
    }
    *headers_start = '\0';
    
    // Body starts after headers
    parsed->body = (char*)current;
    parsed->body_length = strlen(current);
    
    return 0;
}

// Resolve hostname to IP address
int resolve_hostname(const char* hostname, char* ip_str) {
    struct hostent* host_entry = gethostbyname(hostname);
    if (host_entry == NULL) {
        return -1;
    }
    
    struct in_addr addr;
    memcpy(&addr, host_entry->h_addr_list[0], host_entry->h_length);
    strcpy(ip_str, inet_ntoa(addr));
    return 0;
}

// Perform HTTP request
int http_request(const char* url, const char* method, const char* body) {
    char host[256], path[256], ip_str[16];
    int port;
    
    // Parse URL
    if (parse_url(url, host, path, &port) != 0) {
        printf("Error: Invalid URL\n");
        return -1;
    }
    
    printf("=== HTTP Request Details ===\n");
    printf("Host: %s\n", host);
    printf("Path: %s\n", path);
    printf("Port: %d\n", port);
    
    // Resolve hostname
    if (resolve_hostname(host, ip_str) != 0) {
        printf("Error: Cannot resolve hostname %s\n", host);
        return -1;
    }
    printf("Resolved IP: %s\n", ip_str);
    
    // Create socket
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        return -1;
    }
    
    // Connect to server
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    inet_pton(AF_INET, ip_str, &server_addr.sin_addr);
    
    printf("Connecting to %s:%d...\n", ip_str, port);
    if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connection failed");
        close(sockfd);
        return -1;
    }
    printf("Connected successfully!\n\n");
    
    // Build HTTP request
    http_request_t req;
    strcpy(req.method, method);
    strcpy(req.path, path);
    strcpy(req.host, host);
    strcpy(req.version, "HTTP/1.1");
    strcpy(req.headers, "");
    strcpy(req.body, body ? body : "");
    
    char request_buffer[4096];
    int request_len = build_http_request(&req, request_buffer, sizeof(request_buffer));
    
    printf("=== Sending HTTP Request ===\n");
    printf("%s", request_buffer);
    printf("Request size: %d bytes\n\n", request_len);
    
    // Send request
    if (send(sockfd, request_buffer, request_len, 0) < 0) {
        perror("Send failed");
        close(sockfd);
        return -1;
    }
    
    // Receive response
    char response_buffer[8192];
    int total_received = 0;
    int bytes_received;
    
    printf("=== Receiving HTTP Response ===\n");
    while ((bytes_received = recv(sockfd, response_buffer + total_received, 
                                sizeof(response_buffer) - total_received - 1, 0)) > 0) {
        total_received += bytes_received;
        if (total_received >= sizeof(response_buffer) - 1) break;
    }
    
    response_buffer[total_received] = '\0';
    printf("Total received: %d bytes\n\n", total_received);
    
    // Parse and display response
    http_response_t response;
    if (parse_http_response(response_buffer, &response) == 0) {
        printf("=== HTTP Response Analysis ===\n");
        printf("Version: %s\n", response.version);
        printf("Status: %d %s\n", response.status_code, response.status_text);
        printf("Headers:\n%s\n", response.headers);
        printf("Body length: %zu bytes\n", response.body_length);
        
        if (response.body_length > 0) {
            printf("\n=== Response Body ===\n");
            printf("%.500s", response.body);  // Show first 500 chars
            if (response.body_length > 500) {
                printf("\n... (truncated, %zu more bytes)\n", response.body_length - 500);
            }
        }
    }
    
    close(sockfd);
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <URL> [method] [body]\n", argv[0]);
        printf("Examples:\n");
        printf("  %s http://httpbin.org/get\n", argv[0]);
        printf("  %s http://httpbin.org/post POST \"key=value&name=test\"\n", argv[0]);
        return 1;
    }
    
    const char* url = argv[1];
    const char* method = (argc > 2) ? argv[2] : "GET";
    const char* body = (argc > 3) ? argv[3] : NULL;
    
    return http_request(url, method, body);
}
```

### HTTP Status Codes

| Code | Category | Meaning | Example |
|------|----------|---------|---------|
| 1xx | Informational | Request received, continuing | 100 Continue |
| 2xx | Success | Request successful | 200 OK, 201 Created |
| 3xx | Redirection | Further action needed | 301 Moved Permanently |
| 4xx | Client Error | Client error | 404 Not Found, 401 Unauthorized |
| 5xx | Server Error | Server error | 500 Internal Server Error |

### Common HTTP Headers

**Request Headers:**
- `Host`: Target server hostname
- `User-Agent`: Client application identifier
- `Accept`: Acceptable response content types
- `Authorization`: Authentication credentials
- `Content-Type`: Type of request body content
- `Content-Length`: Size of request body

**Response Headers:**
- `Content-Type`: Type of response body content
- `Content-Length`: Size of response body
- `Cache-Control`: Caching directives
- `Set-Cookie`: Cookie data to store
- `Location`: Redirect target URL
- `Server`: Server software identifier

## 3. Simple Mail Transfer Protocol (SMTP)

### Overview
SMTP is used for sending email messages between servers. It operates on port 25 (standard) or 587 (submission) and uses a command-response model.

### SMTP Command Sequence
```
1. HELO/EHLO - Identify client to server
2. MAIL FROM - Specify sender
3. RCPT TO - Specify recipient(s)
4. DATA - Begin message content
5. . (dot) - End message content
6. QUIT - Close connection
```

### SMTP Client Implementation

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

#define SMTP_PORT 587
#define BUFFER_SIZE 1024

typedef struct {
    char server[256];
    int port;
    char from[256];
    char to[256];
    char subject[256];
    char body[2048];
} email_t;

// Send SMTP command and read response
int smtp_command(int sockfd, const char* command, char* response) {
    printf(">> %s", command);
    
    if (send(sockfd, command, strlen(command), 0) < 0) {
        perror("Send failed");
        return -1;
    }
    
    int bytes = recv(sockfd, response, BUFFER_SIZE - 1, 0);
    if (bytes < 0) {
        perror("Receive failed");
        return -1;
    }
    
    response[bytes] = '\0';
    printf("<< %s", response);
    
    // Extract response code
    return atoi(response);
}

// Send email via SMTP
int send_email(const email_t* email) {
    int sockfd;
    struct sockaddr_in server_addr;
    char command[BUFFER_SIZE];
    char response[BUFFER_SIZE];
    int response_code;
    
    printf("=== SMTP Email Client ===\n");
    printf("Server: %s:%d\n", email->server, email->port);
    printf("From: %s\n", email->from);
    printf("To: %s\n", email->to);
    printf("Subject: %s\n\n", email->subject);
    
    // Resolve server hostname
    struct hostent* host_entry = gethostbyname(email->server);
    if (host_entry == NULL) {
        printf("Error: Cannot resolve SMTP server %s\n", email->server);
        return -1;
    }
    
    // Create socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        return -1;
    }
    
    // Connect to SMTP server
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(email->port);
    memcpy(&server_addr.sin_addr, host_entry->h_addr_list[0], host_entry->h_length);
    
    printf("Connecting to SMTP server...\n");
    if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connection failed");
        close(sockfd);
        return -1;
    }
    
    // Read initial server greeting
    recv(sockfd, response, BUFFER_SIZE - 1, 0);
    response[BUFFER_SIZE - 1] = '\0';
    printf("<< %s", response);
    
    // EHLO command
    snprintf(command, sizeof(command), "EHLO localhost\r\n");
    response_code = smtp_command(sockfd, command, response);
    if (response_code != 250) {
        printf("EHLO failed\n");
        close(sockfd);
        return -1;
    }
    
    // MAIL FROM command
    snprintf(command, sizeof(command), "MAIL FROM:<%s>\r\n", email->from);
    response_code = smtp_command(sockfd, command, response);
    if (response_code != 250) {
        printf("MAIL FROM failed\n");
        close(sockfd);
        return -1;
    }
    
    // RCPT TO command
    snprintf(command, sizeof(command), "RCPT TO:<%s>\r\n", email->to);
    response_code = smtp_command(sockfd, command, response);
    if (response_code != 250) {
        printf("RCPT TO failed\n");
        close(sockfd);
        return -1;
    }
    
    // DATA command
    snprintf(command, sizeof(command), "DATA\r\n");
    response_code = smtp_command(sockfd, command, response);
    if (response_code != 354) {
        printf("DATA command failed\n");
        close(sockfd);
        return -1;
    }
    
    // Send email headers and body
    snprintf(command, sizeof(command), 
             "From: %s\r\n"
             "To: %s\r\n"
             "Subject: %s\r\n"
             "Date: %s\r\n"
             "\r\n"
             "%s\r\n"
             ".\r\n",
             email->from, email->to, email->subject, 
             "Wed, 30 Jun 2025 12:00:00 +0000", email->body);
    
    response_code = smtp_command(sockfd, command, response);
    if (response_code != 250) {
        printf("Email sending failed\n");
        close(sockfd);
        return -1;
    }
    
    // QUIT command
    snprintf(command, sizeof(command), "QUIT\r\n");
    smtp_command(sockfd, command, response);
    
    close(sockfd);
    printf("Email sent successfully!\n");
    return 0;
}

int main() {
    email_t email = {
        .server = "smtp.gmail.com",  // Note: Gmail requires authentication
        .port = 587,
        .from = "sender@example.com",
        .to = "recipient@example.com",
        .subject = "Test Email from C Program",
        .body = "Hello,\n\nThis is a test email sent from a C program using SMTP.\n\nBest regards,\nC Program"
    };
    
    printf("Note: This is a basic SMTP client demonstration.\n");
    printf("Real email servers require authentication (SASL/AUTH).\n\n");
    
    return send_email(&email);
}
```

### SMTP Response Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 220 | Service ready | Server ready to proceed |
| 221 | Service closing | Server closing connection |
| 250 | OK | Command completed successfully |
| 354 | Start mail input | Begin sending message content |
| 421 | Service not available | Server temporarily unavailable |
| 450 | Mailbox busy | Mailbox temporarily unavailable |
| 550 | Mailbox unavailable | Mailbox not found or access denied |

## 4. File Transfer Protocol (FTP)

### Overview
FTP is used for transferring files between client and server. It uses two connections: control (port 21) for commands and data (port 20 or dynamic) for file transfer.

### FTP Connection Types
- **Active Mode**: Server initiates data connection to client
- **Passive Mode**: Client initiates data connection to server

### FTP Client Implementation

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

#define FTP_PORT 21
#define BUFFER_SIZE 1024

typedef struct {
    int control_sock;
    char server[256];
    char username[64];
    char password[64];
} ftp_client_t;

// Send FTP command and read response
int ftp_command(ftp_client_t* client, const char* command, char* response) {
    printf(">> %s", command);
    
    if (send(client->control_sock, command, strlen(command), 0) < 0) {
        perror("Send failed");
        return -1;
    }
    
    int bytes = recv(client->control_sock, response, BUFFER_SIZE - 1, 0);
    if (bytes <= 0) {
        perror("Receive failed");
        return -1;
    }
    
    response[bytes] = '\0';
    printf("<< %s", response);
    
    return atoi(response);
}

// Connect to FTP server
int ftp_connect(ftp_client_t* client, const char* server) {
    struct sockaddr_in server_addr;
    struct hostent* host_entry;
    char response[BUFFER_SIZE];
    
    strcpy(client->server, server);
    
    // Resolve hostname
    host_entry = gethostbyname(server);
    if (host_entry == NULL) {
        printf("Error: Cannot resolve FTP server %s\n", server);
        return -1;
    }
    
    // Create control socket
    client->control_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (client->control_sock < 0) {
        perror("Socket creation failed");
        return -1;
    }
    
    // Connect to server
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(FTP_PORT);
    memcpy(&server_addr.sin_addr, host_entry->h_addr_list[0], host_entry->h_length);
    
    printf("Connecting to FTP server %s...\n", server);
    if (connect(client->control_sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connection failed");
        close(client->control_sock);
        return -1;
    }
    
    // Read server greeting
    recv(client->control_sock, response, BUFFER_SIZE - 1, 0);
    response[BUFFER_SIZE - 1] = '\0';
    printf("<< %s", response);
    
    return 0;
}

// Login to FTP server
int ftp_login(ftp_client_t* client, const char* username, const char* password) {
    char command[256];
    char response[BUFFER_SIZE];
    int response_code;
    
    strcpy(client->username, username);
    strcpy(client->password, password);
    
    // Send username
    snprintf(command, sizeof(command), "USER %s\r\n", username);
    response_code = ftp_command(client, command, response);
    
    if (response_code == 331) {  // Need password
        snprintf(command, sizeof(command), "PASS %s\r\n", password);
        response_code = ftp_command(client, command, response);
    }
    
    if (response_code == 230) {  // Login successful
        printf("Login successful!\n");
        return 0;
    } else {
        printf("Login failed (code: %d)\n", response_code);
        return -1;
    }
}

// Enter passive mode and get data connection info
int ftp_passive_mode(ftp_client_t* client, char* data_ip, int* data_port) {
    char command[64];
    char response[BUFFER_SIZE];
    int response_code;
    
    snprintf(command, sizeof(command), "PASV\r\n");
    response_code = ftp_command(client, command, response);
    
    if (response_code != 227) {
        printf("Passive mode failed\n");
        return -1;
    }
    
    // Parse PASV response: 227 Entering Passive Mode (h1,h2,h3,h4,p1,p2)
    char* start = strchr(response, '(');
    if (!start) return -1;
    
    int h1, h2, h3, h4, p1, p2;
    if (sscanf(start + 1, "%d,%d,%d,%d,%d,%d", &h1, &h2, &h3, &h4, &p1, &p2) != 6) {
        return -1;
    }
    
    snprintf(data_ip, 16, "%d.%d.%d.%d", h1, h2, h3, h4);
    *data_port = (p1 << 8) | p2;
    
    printf("Passive mode: %s:%d\n", data_ip, *data_port);
    return 0;
}

// List directory contents
int ftp_list(ftp_client_t* client) {
    char data_ip[16];
    int data_port;
    int data_sock;
    struct sockaddr_in data_addr;
    char command[64];
    char response[BUFFER_SIZE];
    char buffer[BUFFER_SIZE];
    int response_code, bytes;
    
    // Enter passive mode
    if (ftp_passive_mode(client, data_ip, &data_port) != 0) {
        return -1;
    }
    
    // Create data connection
    data_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (data_sock < 0) {
        perror("Data socket creation failed");
        return -1;
    }
    
    memset(&data_addr, 0, sizeof(data_addr));
    data_addr.sin_family = AF_INET;
    data_addr.sin_port = htons(data_port);
    inet_pton(AF_INET, data_ip, &data_addr.sin_addr);
    
    if (connect(data_sock, (struct sockaddr*)&data_addr, sizeof(data_addr)) < 0) {
        perror("Data connection failed");
        close(data_sock);
        return -1;
    }
    
    // Send LIST command
    snprintf(command, sizeof(command), "LIST\r\n");
    response_code = ftp_command(client, command, response);
    
    if (response_code == 150 || response_code == 125) {  // Data connection opened
        printf("\n=== Directory Listing ===\n");
        while ((bytes = recv(data_sock, buffer, BUFFER_SIZE - 1, 0)) > 0) {
            buffer[bytes] = '\0';
            printf("%s", buffer);
        }
        printf("\n========================\n");
    }
    
    close(data_sock);
    
    // Read completion response
    recv(client->control_sock, response, BUFFER_SIZE - 1, 0);
    response[BUFFER_SIZE - 1] = '\0';
    printf("<< %s", response);
    
    return 0;
}

// Change directory
int ftp_cwd(ftp_client_t* client, const char* directory) {
    char command[256];
    char response[BUFFER_SIZE];
    int response_code;
    
    snprintf(command, sizeof(command), "CWD %s\r\n", directory);
    response_code = ftp_command(client, command, response);
    
    if (response_code == 250) {
        printf("Directory changed successfully\n");
        return 0;
    } else {
        printf("Failed to change directory\n");
        return -1;
    }
}

// Print working directory
int ftp_pwd(ftp_client_t* client) {
    char command[64];
    char response[BUFFER_SIZE];
    int response_code;
    
    snprintf(command, sizeof(command), "PWD\r\n");
    response_code = ftp_command(client, command, response);
    
    return (response_code == 257) ? 0 : -1;
}

// Disconnect from FTP server
void ftp_disconnect(ftp_client_t* client) {
    char command[64];
    char response[BUFFER_SIZE];
    
    snprintf(command, sizeof(command), "QUIT\r\n");
    ftp_command(client, command, response);
    
    close(client->control_sock);
    printf("Disconnected from FTP server\n");
}

int main() {
    ftp_client_t client;
    
    printf("=== Simple FTP Client ===\n");
    
    // Connect to server
    if (ftp_connect(&client, "ftp.example.com") != 0) {
        return 1;
    }
    
    // Login (many servers allow anonymous login)
    if (ftp_login(&client, "anonymous", "user@example.com") != 0) {
        ftp_disconnect(&client);
        return 1;
    }
    
    // Print working directory
    ftp_pwd(&client);
    
    // List directory contents
    ftp_list(&client);
    
    // Change to pub directory (common on FTP servers)
    ftp_cwd(&client, "pub");
    ftp_list(&client);
    
    // Disconnect
    ftp_disconnect(&client);
    
    return 0;
}
```

### FTP Response Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 150 | File status okay | About to open data connection |
| 200 | Command okay | Command successful |
| 220 | Service ready | Server ready for new user |
| 226 | Closing data connection | File transfer complete |
| 230 | User logged in | Login successful |
| 331 | User name okay | Need password |
| 425 | Can't open data connection | Data connection failed |
| 530 | Not logged in | Authentication required |

## 5. Dynamic Host Configuration Protocol (DHCP)

### Overview
DHCP automatically assigns IP addresses and network configuration to devices on a network. It operates using a client-server model over UDP ports 67 (server) and 68 (client).

### DHCP Message Flow (DORA Process)
```
1. DISCOVER - Client broadcasts request for IP
2. OFFER    - Server offers IP configuration
3. REQUEST  - Client requests specific configuration
4. ACK      - Server acknowledges and assigns IP
```

### DHCP Message Structure

```c
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#define DHCP_SERVER_PORT 67
#define DHCP_CLIENT_PORT 68
#define DHCP_MAGIC_COOKIE 0x63825363

// DHCP Message Types
#define DHCP_DISCOVER 1
#define DHCP_OFFER    2
#define DHCP_REQUEST  3
#define DHCP_DECLINE  4
#define DHCP_ACK      5
#define DHCP_NAK      6
#define DHCP_RELEASE  7
#define DHCP_INFORM   8

// DHCP Options
#define DHCP_OPT_SUBNET_MASK        1
#define DHCP_OPT_ROUTER             3
#define DHCP_OPT_DNS_SERVER         6
#define DHCP_OPT_REQUESTED_IP       50
#define DHCP_OPT_LEASE_TIME         51
#define DHCP_OPT_MESSAGE_TYPE       53
#define DHCP_OPT_SERVER_ID          54
#define DHCP_OPT_RENEWAL_TIME       58
#define DHCP_OPT_REBINDING_TIME     59
#define DHCP_OPT_END                255

// DHCP Message Structure
typedef struct {
    uint8_t  op;        // Message op code / message type (1=request, 2=reply)
    uint8_t  htype;     // Hardware address type (1=Ethernet)
    uint8_t  hlen;      // Hardware address length (6 for Ethernet)
    uint8_t  hops;      // Hops
    uint32_t xid;       // Transaction ID
    uint16_t secs;      // Seconds elapsed since client began address acquisition
    uint16_t flags;     // Flags (broadcast flag in bit 15)
    uint32_t ciaddr;    // Client IP address
    uint32_t yiaddr;    // 'your' (client) IP address
    uint32_t siaddr;    // Next server IP address
    uint32_t giaddr;    // Relay agent IP address
    uint8_t  chaddr[16]; // Client hardware address
    uint8_t  sname[64]; // Optional server host name
    uint8_t  file[128]; // Boot file name
    uint32_t magic;     // Magic cookie (0x63825363)
    uint8_t  options[312]; // Optional parameters field
} __attribute__((packed)) dhcp_message_t;

// Add DHCP option to message
int add_dhcp_option(uint8_t* options, int* offset, uint8_t type, uint8_t length, const void* data) {
    options[(*offset)++] = type;
    options[(*offset)++] = length;
    memcpy(&options[*offset], data, length);
    *offset += length;
    return 0;
}

// Create DHCP DISCOVER message
int create_dhcp_discover(dhcp_message_t* msg, uint32_t xid, const uint8_t* mac_addr) {
    memset(msg, 0, sizeof(dhcp_message_t));
    
    // Basic fields
    msg->op = 1;        // BOOTREQUEST
    msg->htype = 1;     // Ethernet
    msg->hlen = 6;      // Ethernet MAC length
    msg->hops = 0;
    msg->xid = htonl(xid);
    msg->secs = 0;
    msg->flags = htons(0x8000);  // Broadcast flag
    msg->ciaddr = 0;    // Client IP (unknown)
    msg->yiaddr = 0;    // Your IP (to be assigned)
    msg->siaddr = 0;    // Server IP (unknown)
    msg->giaddr = 0;    // Gateway IP (none)
    
    // Hardware address
    memcpy(msg->chaddr, mac_addr, 6);
    
    // Magic cookie
    msg->magic = htonl(DHCP_MAGIC_COOKIE);
    
    // DHCP options
    int offset = 0;
    
    // Message type option
    uint8_t msg_type = DHCP_DISCOVER;
    add_dhcp_option(msg->options, &offset, DHCP_OPT_MESSAGE_TYPE, 1, &msg_type);
    
    // End option
    msg->options[offset++] = DHCP_OPT_END;
    
    return 0;
}

// Parse DHCP options
void parse_dhcp_options(const uint8_t* options, int length) {
    int i = 0;
    
    printf("DHCP Options:\n");
    while (i < length && options[i] != DHCP_OPT_END) {
        uint8_t type = options[i++];
        
        if (type == 0) continue; // Padding
        
        uint8_t len = options[i++];
        
        switch (type) {
            case DHCP_OPT_MESSAGE_TYPE:
                printf("  Message Type: %d (", options[i]);
                switch (options[i]) {
                    case DHCP_DISCOVER: printf("DISCOVER"); break;
                    case DHCP_OFFER:    printf("OFFER"); break;
                    case DHCP_REQUEST:  printf("REQUEST"); break;
                    case DHCP_ACK:      printf("ACK"); break;
                    case DHCP_NAK:      printf("NAK"); break;
                    default:            printf("Unknown"); break;
                }
                printf(")\n");
                break;
                
            case DHCP_OPT_SUBNET_MASK:
                if (len == 4) {
                    struct in_addr addr;
                    memcpy(&addr, &options[i], 4);
                    printf("  Subnet Mask: %s\n", inet_ntoa(addr));
                }
                break;
                
            case DHCP_OPT_ROUTER:
                if (len >= 4) {
                    struct in_addr addr;
                    memcpy(&addr, &options[i], 4);
                    printf("  Router: %s\n", inet_ntoa(addr));
                }
                break;
                
            case DHCP_OPT_DNS_SERVER:
                printf("  DNS Servers: ");
                for (int j = 0; j < len; j += 4) {
                    struct in_addr addr;
                    memcpy(&addr, &options[i + j], 4);
                    printf("%s ", inet_ntoa(addr));
                }
                printf("\n");
                break;
                
            case DHCP_OPT_LEASE_TIME:
                if (len == 4) {
                    uint32_t lease_time = ntohl(*(uint32_t*)&options[i]);
                    printf("  Lease Time: %u seconds (%u hours)\n", 
                           lease_time, lease_time / 3600);
                }
                break;
                
            case DHCP_OPT_SERVER_ID:
                if (len == 4) {
                    struct in_addr addr;
                    memcpy(&addr, &options[i], 4);
                    printf("  DHCP Server: %s\n", inet_ntoa(addr));
                }
                break;
                
            default:
                printf("  Option %d: %d bytes\n", type, len);
                break;
        }
        
        i += len;
    }
}

// Parse DHCP message
void parse_dhcp_message(const dhcp_message_t* msg) {
    printf("=== DHCP Message Analysis ===\n");
    printf("Operation: %s\n", (msg->op == 1) ? "BOOTREQUEST" : "BOOTREPLY");
    printf("Hardware Type: %d (Ethernet)\n", msg->htype);
    printf("Hardware Length: %d\n", msg->hlen);
    printf("Transaction ID: 0x%08x\n", ntohl(msg->xid));
    printf("Seconds: %d\n", ntohs(msg->secs));
    printf("Flags: 0x%04x%s\n", ntohs(msg->flags), 
           (ntohs(msg->flags) & 0x8000) ? " (Broadcast)" : "");
    
    if (msg->ciaddr != 0) {
        struct in_addr addr = { .s_addr = msg->ciaddr };
        printf("Client IP: %s\n", inet_ntoa(addr));
    }
    
    if (msg->yiaddr != 0) {
        struct in_addr addr = { .s_addr = msg->yiaddr };
        printf("Your IP: %s\n", inet_ntoa(addr));
    }
    
    if (msg->siaddr != 0) {
        struct in_addr addr = { .s_addr = msg->siaddr };
        printf("Server IP: %s\n", inet_ntoa(addr));
    }
    
    printf("Client MAC: %02x:%02x:%02x:%02x:%02x:%02x\n",
           msg->chaddr[0], msg->chaddr[1], msg->chaddr[2],
           msg->chaddr[3], msg->chaddr[4], msg->chaddr[5]);
    
    printf("Magic Cookie: 0x%08x\n", ntohl(msg->magic));
    
    if (ntohl(msg->magic) == DHCP_MAGIC_COOKIE) {
        parse_dhcp_options(msg->options, sizeof(msg->options));
    }
}

// Simple DHCP client demonstration
int dhcp_discover_demo() {
    dhcp_message_t discover_msg;
    uint8_t mac_addr[6] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55};
    uint32_t xid = 0x12345678;
    
    printf("=== DHCP DISCOVER Message Creation ===\n");
    
    create_dhcp_discover(&discover_msg, xid, mac_addr);
    
    printf("Created DHCP DISCOVER message:\n");
    printf("Message size: %zu bytes\n", sizeof(dhcp_message_t));
    
    parse_dhcp_message(&discover_msg);
    
    printf("\nNote: This is a demonstration of DHCP message structure.\n");
    printf("A real DHCP client would broadcast this message and listen for OFFER responses.\n");
    
    return 0;
}

int main() {
    printf("=== DHCP Protocol Demonstration ===\n\n");
    
    printf("DHCP Message Flow (DORA Process):\n");
    printf("1. DISCOVER - Client broadcasts: 'I need an IP address'\n");
    printf("2. OFFER    - Server responds: 'Here's an available IP'\n");
    printf("3. REQUEST  - Client requests: 'I want that specific IP'\n");
    printf("4. ACK      - Server confirms: 'IP assigned to you'\n\n");
    
    return dhcp_discover_demo();
}
```

### DHCP Options Summary

| Option | Name | Description |
|--------|------|-------------|
| 1 | Subnet Mask | Network subnet mask |
| 3 | Router | Default gateway IP |
| 6 | DNS Server | DNS server IP addresses |
| 15 | Domain Name | DNS domain name |
| 51 | Lease Time | IP address lease duration |
| 53 | Message Type | DHCP message type |
| 54 | Server Identifier | DHCP server IP address |

## Practical Exercises and Labs

### Exercise 1: Protocol Analysis
Write programs to:
1. Capture and analyze DNS queries using raw sockets
2. Parse HTTP headers from real web requests
3. Monitor DHCP traffic on your network

### Exercise 2: Protocol Implementation
Implement simplified versions of:
1. A DNS resolver that queries multiple DNS servers
2. An HTTP proxy server
3. A basic FTP client with file download capability

### Exercise 3: Protocol Security
Investigate and demonstrate:
1. DNS cache poisoning vulnerabilities
2. HTTP header injection attacks
3. SMTP relay abuse prevention

## Debugging and Testing Tools

### Network Analysis Tools
```bash
# Capture DNS traffic
tcpdump -i any port 53

# Analyze HTTP traffic
wireshark -k -i any -f "port 80"

# Test SMTP connection
telnet smtp.gmail.com 587

# Check DNS resolution
nslookup google.com
dig google.com

# Monitor DHCP traffic
tcpdump -i any port 67 or port 68
```

### Development Tools
```bash
# Compile networking programs
gcc -o dns_client dns_client.c
gcc -o http_client http_client.c -lresolv

# Test with various servers
./dns_client google.com
./http_client http://httpbin.org/get
```

## Study Materials and References

### Essential Reading
- **RFC Documents:**
  - RFC 1035 (DNS)
  - RFC 2616 (HTTP/1.1), RFC 7540 (HTTP/2)
  - RFC 5321 (SMTP)
  - RFC 959 (FTP)
  - RFC 2131 (DHCP)

### Recommended Books
- "TCP/IP Illustrated, Volume 1" by W. Richard Stevens
- "Computer Networks" by Andrew Tanenbaum
- "Unix Network Programming" by W. Richard Stevens

### Online Resources
- [Wireshark Network Analysis](https://www.wireshark.org/docs/)
- [RFC Editor](https://www.rfc-editor.org/)
- [Network Programming Tutorials](https://beej.us/guide/bgnet/)

### Practice Platforms
- [HTTPBin](http://httpbin.org/) - HTTP testing service
- [Packet Tracer](https://www.netacad.com/courses/packet-tracer) - Network simulation
- [GNS3](https://www.gns3.com/) - Network emulation


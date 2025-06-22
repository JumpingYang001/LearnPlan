# HTTP Fundamentals

## Overview
Covers the basics of HTTP including client-server architecture, statelessness, request-response cycle, HTTP versions, and URI structure.

## Key Concepts
- Client-server architecture
- Stateless nature of HTTP
- Request-response cycle
- HTTP versions overview (HTTP/1.0, HTTP/1.1, HTTP/2, HTTP/3)
- URI structure and components

## C/C++ Example: Simple HTTP GET Request (using sockets)
```c
// Simple HTTP GET request using C sockets (POSIX)
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

int main() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in server;
    server.sin_addr.s_addr = inet_addr("93.184.216.34"); // example.com
    server.sin_family = AF_INET;
    server.sin_port = htons(80);
    connect(sock, (struct sockaddr *)&server, sizeof(server));
    char *message = "GET / HTTP/1.1\r\nHost: example.com\r\nConnection: close\r\n\r\n";
    send(sock, message, strlen(message), 0);
    char response[4096];
    int len;
    while ((len = recv(sock, response, sizeof(response)-1, 0)) > 0) {
        response[len] = '\0';
        printf("%s", response);
    }
    close(sock);
    return 0;
}
```

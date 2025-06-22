# Project: Simple HTTP Server

## Description
Create a basic HTTP/1.1 server in C/C++. Support multiple concurrent clients and implement common request methods.

## C/C++ Example: Minimal HTTP Server (single-threaded)
```c
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

int main() {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(8080);
    bind(server_fd, (struct sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 5);
    while (1) {
        int client = accept(server_fd, NULL, NULL);
        char buffer[1024];
        read(client, buffer, sizeof(buffer));
        char *response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nHello, World!";
        write(client, response, strlen(response));
        close(client);
    }
    return 0;
}
```

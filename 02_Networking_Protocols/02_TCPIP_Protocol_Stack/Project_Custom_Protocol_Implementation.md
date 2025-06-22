# Project: Custom Protocol Implementation

## Objective
Design and implement a simple application protocol over TCP. Document the protocol specification.

## Example Code (Python)
```python
# Server
import socket
s = socket.socket()
s.bind(('0.0.0.0', 12345))
s.listen(1)
conn, addr = s.accept()
data = conn.recv(1024)
print(f"Received: {data.decode()}")
conn.send(b'ACK')
conn.close()

# Client
import socket
s = socket.socket()
s.connect(('127.0.0.1', 12345))
s.send(b'HELLO')
print(s.recv(1024))
s.close()
```
## Example Code (C++)
### Server
```cpp
#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

int main() {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(12345);
    bind(server_fd, (sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 1);
    int client_fd = accept(server_fd, nullptr, nullptr);
    char buffer[1024] = {0};
    read(client_fd, buffer, 1024);
    std::cout << "Received: " << buffer << std::endl;
    send(client_fd, "ACK", 3, 0);
    close(client_fd);
    close(server_fd);
    return 0;
}
```
### Client
```cpp
#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

int main() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(12345);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    connect(sock, (sockaddr*)&addr, sizeof(addr));
    send(sock, "HELLO", 5, 0);
    char buffer[1024] = {0};
    read(sock, buffer, 1024);
    std::cout << buffer << std::endl;
    close(sock);
    return 0;
}
```
*Note: Works on Linux. Compile with `g++ server.cpp -o server` and `g++ client.cpp -o client`.*

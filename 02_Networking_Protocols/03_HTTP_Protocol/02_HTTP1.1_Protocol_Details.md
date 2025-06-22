# HTTP/1.1 Protocol Details

## Overview
Detailed explanation of HTTP/1.1 message format, methods, status codes, and headers.

## Message Format
- Request line and response status line
- Headers
- Message body

## C/C++ Example: Parsing HTTP Response
```c++
// Parse HTTP response headers in C++
#include <iostream>
#include <sstream>
#include <string>

int main() {
    std::string response = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 13\r\n\r\nHello, world!";
    std::istringstream resp_stream(response);
    std::string line;
    while (std::getline(resp_stream, line) && line != "\r") {
        std::cout << line << std::endl;
    }
    return 0;
}
```

## Request Methods
- GET, POST, PUT, DELETE, HEAD, OPTIONS, PATCH, TRACE, CONNECT
- Safe and idempotent methods

## Status Codes
- 1xx: Informational
- 2xx: Success
- 3xx: Redirection
- 4xx: Client Error
- 5xx: Server Error

## Common Headers
- General, Request, Response, Entity, Custom headers

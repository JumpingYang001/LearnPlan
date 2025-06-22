# HTTP/1.1 Features

## Overview
Explains persistent connections, pipelining, chunked transfer encoding, byte ranges, content negotiation, caching, compression, authentication, and cookies.

## C/C++ Example: Chunked Transfer Encoding (Server Side)
```c++
// Minimal HTTP chunked response in C++
#include <iostream>
#include <string>
int main() {
    std::cout << "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n";
    std::cout << "5\r\nHello\r\n";
    std::cout << "6\r\n World\r\n";
    std::cout << "0\r\n\r\n";
    return 0;
}
```

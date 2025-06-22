# Project: HTTP Protocol Analyzer

## Description
Implement a tool to parse and display HTTP messages. Support HTTP/1.1 and HTTP/2 formats.

## C/C++ Example: Simple HTTP/1.1 Message Parser
```c
#include <stdio.h>
#include <string.h>

void parse_http(const char *msg) {
    const char *line = msg;
    while (*line) {
        const char *next = strstr(line, "\r\n");
        if (!next || next == line) break;
        printf("%.*s\n", (int)(next - line), line);
        line = next + 2;
    }
}

int main() {
    const char *http_msg = "GET / HTTP/1.1\r\nHost: example.com\r\n\r\n";
    parse_http(http_msg);
    return 0;
}
```

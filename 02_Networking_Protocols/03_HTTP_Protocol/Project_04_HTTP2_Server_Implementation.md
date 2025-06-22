# Project: HTTP/2 Server Implementation

## Description
Extend the HTTP server to support HTTP/2. Implement stream multiplexing and header compression.

## C/C++ Example: HTTP/2 Server (nghttp2 pseudo-code)
```c++
// Requires nghttp2 library
#include <nghttp2/nghttp2.h>
// ... setup server, accept connections ...
// Use nghttp2_session_recv/send for HTTP/2 frames
// Implement stream multiplexing logic
```

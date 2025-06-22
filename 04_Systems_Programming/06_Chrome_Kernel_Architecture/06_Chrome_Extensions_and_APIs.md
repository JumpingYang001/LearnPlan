# Chrome Extensions and APIs

## Overview
Chrome extensions enhance browser functionality using content scripts, background pages, and extension APIs.

## Key Concepts
- Extension architecture
- Content scripts and background pages
- Extension API usage

## Example: Extension-like Message Passing in C++
```cpp
#include <iostream>
#include <string>

void sendMessage(const std::string& msg) {
    std::cout << "Message sent: " << msg << std::endl;
}

int main() {
    sendMessage("Hello from extension!");
    return 0;
}
```

This code simulates message passing, similar to Chrome extension APIs.

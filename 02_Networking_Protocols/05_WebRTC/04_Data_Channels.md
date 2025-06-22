# Data Channels

## Overview
Learn about RTCDataChannel API, reliable/unreliable data transfer, and use cases like file transfer and messaging.

### C/C++ Example: Sending Data Over a Channel (Pseudocode)
```cpp
// Pseudocode for sending data over a WebRTC data channel
#include <iostream>

void sendData(const std::string& data) {
    // Assume dataChannel is established
    // dataChannel->Send(data);
    std::cout << "Data sent: " << data << std::endl;
}

int main() {
    sendData("Hello, WebRTC!");
    return 0;
}
```

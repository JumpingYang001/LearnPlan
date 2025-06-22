# Project: IP Routing Simulator

## Objective
Implement a basic IP routing algorithm. Demonstrate path selection and forwarding.

## Example Code (C++)
```cpp
#include <iostream>
#include <string>

std::string route_packet(const std::string& dest_ip) {
    if (dest_ip.rfind("192.168.1.", 0) == 0) {
        return "eth0";
    } else if (dest_ip.rfind("10.", 0) == 0) {
        return "eth1";
    } else {
        return "eth2";
    }
}

int main() {
    std::cout << route_packet("10.1.2.3") << std::endl;
    return 0;
}
```

# TCP/IP Fundamentals

## Description
Covers the basics of the TCP/IP protocol stack, OSI model comparison, encapsulation/decapsulation, protocol data units, and header formats.

## Example
- Diagram of TCP/IP vs OSI layers
- Example of encapsulation process

### C++ Example: Encapsulation Process
```cpp
#include <iostream>
#include <string>

std::string encapsulate(const std::string& data) {
    std::string app = "APP[" + data + "]";
    std::string tcp = "TCP[" + app + "]";
    std::string ip = "IP[" + tcp + "]";
    std::string eth = "ETH[" + ip + "]";
    return eth;
}

int main() {
    std::cout << encapsulate("Hello, TCP/IP!") << std::endl;
    return 0;
}
```

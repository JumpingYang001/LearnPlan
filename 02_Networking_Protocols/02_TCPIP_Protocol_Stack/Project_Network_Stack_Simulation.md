# Project: Network Stack Simulation

## Objective
Create a simplified TCP/IP stack simulation. Demonstrate encapsulation and protocol operation.

## Example Code (C++)
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
    std::cout << encapsulate("Hello") << std::endl;
    return 0;
}
```

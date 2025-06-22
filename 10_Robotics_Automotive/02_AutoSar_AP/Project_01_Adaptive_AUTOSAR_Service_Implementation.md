# Project 1: Adaptive AUTOSAR Service Implementation

## Description
Develop a complete service following Adaptive AUTOSAR specifications, including service discovery and communication. Test with multiple service instances and consumers.

## Example Code
```cpp
// Service registration (pseudo-code)
ServiceRegistry.registerService("VehicleStatus", ip, port);
// Service discovery
auto services = ServiceRegistry.findServices("VehicleStatus");
// Communication
for (auto& service : services) {
    service.send("status_request");
}
```

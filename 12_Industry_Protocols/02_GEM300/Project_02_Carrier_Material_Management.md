# Project: Carrier and Material Management System

## Description
Develop a system for tracking carriers and materials, including carrier verification, slot mapping, substrate tracking, and genealogy.

## Example Code
```cpp
// Carrier and Substrate Tracking Example
struct Carrier {
    std::string id;
    std::vector<std::string> slots;
};

Carrier c = {"CARRIER123", {"SLOT1", "SLOT2"}};
```

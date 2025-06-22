# Integration with Classic AUTOSAR

## Description
Understand communication, gateway implementation, data conversion, and integration solutions.

## Example
```cpp
// Gateway pseudo-code
ClassicMsg msg = receiveClassic();
AdaptiveMsg aMsg = convertToAdaptive(msg);
sendToAdaptive(aMsg);
```

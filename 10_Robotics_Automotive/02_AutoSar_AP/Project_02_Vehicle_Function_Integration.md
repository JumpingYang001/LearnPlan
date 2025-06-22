# Project 2: Vehicle Function Integration

## Description
Implement a vehicle function (e.g., automated parking) using Adaptive AUTOSAR. Integrate with sensors and actuators, and implement safety mechanisms and error handling.

## Example Code
```cpp
// Sensor integration (pseudo-code)
SensorData data = sensors.read();
if (data.isValid()) {
    actuators.control("park");
} else {
    logError("Sensor failure");
}
// Safety check
if (!safetyCheck()) {
    actuators.stop();
}
```

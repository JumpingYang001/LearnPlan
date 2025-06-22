# BLE (Bluetooth Low Energy)

## Explanation
BLE is a wireless protocol for short-range, low-power communication. It uses GATT profiles for data exchange and is widely used in wearables and asset tracking.

## Example
```python
# Example: BLE Peripheral (bluepy)
from bluepy.btle import Peripheral
p = Peripheral()
# Setup GATT services and characteristics here
p.disconnect()
```

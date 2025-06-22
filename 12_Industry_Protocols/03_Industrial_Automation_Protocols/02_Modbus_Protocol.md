# Modbus Protocol

## Explanation
Modbus is a widely used industrial protocol for communication between devices. It exists in RTU and TCP forms, uses addressing and function codes, and is simple to implement for both client and server applications.

## Example
```python
# Example: Modbus TCP Client (Python, using pymodbus)
from pymodbus.client.sync import ModbusTcpClient
client = ModbusTcpClient('localhost', port=502)
client.connect()
result = client.read_coils(1, 10)
print(result.bits)
client.close()
```

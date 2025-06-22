# WebSockets for IoT

## Explanation
WebSockets enable full-duplex communication between IoT devices and servers, allowing real-time data exchange. Useful for applications needing instant updates.

## Example
```python
# Example: WebSocket Client
import websocket
ws = websocket.create_connection('ws://echo.websocket.org')
ws.send('Hello IoT')
print(ws.recv())
ws.close()
```

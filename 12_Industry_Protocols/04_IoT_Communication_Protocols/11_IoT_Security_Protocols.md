# IoT Security Protocols

## Explanation
Security is critical in IoT. Protocols like TLS/DTLS, OAuth 2.0, and OpenID Connect are used to secure communication and authenticate devices.

## Example
```python
# Example: Secure MQTT with TLS
import paho.mqtt.client as mqtt
client = mqtt.Client()
client.tls_set()
client.connect('broker.example.com', 8883)
client.publish('secure/topic', 'Secure Message')
client.disconnect()
```

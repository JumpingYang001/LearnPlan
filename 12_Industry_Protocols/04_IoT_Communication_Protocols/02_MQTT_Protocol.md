# MQTT Protocol

## Explanation
MQTT is a lightweight publish-subscribe messaging protocol ideal for IoT due to its low bandwidth and power requirements. It uses topics for message routing and supports different Quality of Service (QoS) levels.

## Example
```python
# Example: Simple MQTT Publisher (Python)
import paho.mqtt.client as mqtt
client = mqtt.Client()
client.connect('broker.hivemq.com', 1883)
client.publish('iot/topic', 'Hello IoT!')
client.disconnect()
```

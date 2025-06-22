# Project: Industrial Telemetry Application

## Description
Build a system to collect data from industrial equipment using MQTT. Implement reliable data delivery with QoS 2 and create alerting and notification systems.

## Example Code (Python, paho-mqtt)
```python
import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc):
    print("Connected with result code ", rc)
    client.subscribe("factory/equipment/+/telemetry")

def on_message(client, userdata, msg):
    print(f"Telemetry: {msg.topic} - {msg.payload.decode()}")
    # Add alerting logic here

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("broker.hivemq.com", 1883, 60)
client.loop_start()

# Publish telemetry with QoS 2
client.publish("factory/equipment/1/telemetry", "status=OK", qos=2)
```

## Alerting
- Integrate with email/SMS APIs for notifications.

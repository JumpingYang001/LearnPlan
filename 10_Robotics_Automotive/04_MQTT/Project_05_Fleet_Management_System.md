# Project: Fleet Management System

## Description
Create a system to track and manage vehicle fleets using MQTT. Implement location tracking, telemetry collection, geofencing, and route optimization features.

## Example Code (Python, paho-mqtt)
```python
import paho.mqtt.client as mqtt
import random, time

def on_connect(client, userdata, flags, rc):
    print("Connected with result code ", rc)
    client.subscribe("fleet/vehicle/+/location")

def on_message(client, userdata, msg):
    print(f"Vehicle update: {msg.topic} - {msg.payload.decode()}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("broker.hivemq.com", 1883, 60)
client.loop_start()

# Simulate vehicle location publishing
for i in range(3):
    lat, lon = 40.0 + random.random(), -74.0 + random.random()
    client.publish("fleet/vehicle/1/location", f"{lat},{lon}", qos=1)
    time.sleep(2)
```

## Geofencing
- Use geospatial libraries to implement geofencing logic.

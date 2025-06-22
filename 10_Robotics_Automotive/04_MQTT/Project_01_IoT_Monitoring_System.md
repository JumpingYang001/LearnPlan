# Project: IoT Monitoring System

## Description
Build a system to monitor multiple IoT sensors using MQTT. Implement different QoS levels for different data types and create visualization dashboards for real-time data.

## Example Code (Python, paho-mqtt)
```python
import paho.mqtt.client as mqtt
import random, time

def on_connect(client, userdata, flags, rc):
    print("Connected with result code ", rc)
    client.subscribe("sensors/+/data")

def on_message(client, userdata, msg):
    print(f"Received from {msg.topic}: {msg.payload.decode()}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("broker.hivemq.com", 1883, 60)
client.loop_start()

# Simulate sensor publishing
for i in range(5):
    temp = random.uniform(20, 30)
    client.publish("sensors/temp/data", f"{temp}", qos=1)
    time.sleep(1)
```

## Dashboard
- Use tools like Node-RED or Grafana to visualize MQTT data in real time.

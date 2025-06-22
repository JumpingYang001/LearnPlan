# MQTT Fundamentals

## Explanation
This section covers the basics of the MQTT protocol, including the publish-subscribe pattern, architecture, protocol versions, and Quality of Service (QoS) levels.

### Publish-Subscribe Pattern
MQTT uses a publish-subscribe model where clients publish messages to topics and other clients subscribe to those topics to receive messages.

#### Example (Python, using paho-mqtt):
```python
import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc):
    print("Connected with result code ", rc)
    client.subscribe("test/topic")

def on_message(client, userdata, msg):
    print(msg.topic, msg.payload.decode())

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("broker.hivemq.com", 1883, 60)
client.loop_start()
client.publish("test/topic", "Hello MQTT!")
```

---

### MQTT Architecture
- Broker: Central server that routes messages
- Publisher: Sends messages
- Subscriber: Receives messages

### Protocol Versions
- MQTT 3.1.1
- MQTT 5.0

### Quality of Service (QoS)
- 0: At most once
- 1: At least once
- 2: Exactly once

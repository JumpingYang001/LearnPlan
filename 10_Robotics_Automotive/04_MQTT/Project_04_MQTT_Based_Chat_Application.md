# Project: MQTT-Based Chat Application

## Description
Develop a simple chat application using MQTT. Implement presence detection, offline messaging, user authentication, and private messaging.

## Example Code (Python, paho-mqtt)
```python
import paho.mqtt.client as mqtt

username = "user1"
chat_topic = f"chat/{username}"

client = mqtt.Client()
client.username_pw_set(username, "password")
client.connect("broker.hivemq.com", 1883, 60)

client.loop_start()

# Presence detection
client.publish("chat/presence", f"{username} online", retain=True)

# Send a message
client.publish("chat/user2", "Hello, user2!")

# Subscribe to own chat topic
client.subscribe(chat_topic)

def on_message(client, userdata, msg):
    print(f"Message: {msg.payload.decode()}")

client.on_message = on_message
```

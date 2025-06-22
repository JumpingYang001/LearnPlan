# MQTT Topics and Messages

## Explanation
This section explains topic structures, wildcards, message formats, retained messages, and topic design patterns.

### Topic Structure & Wildcards
- Topics are hierarchical, e.g., `home/livingroom/temperature`
- `+` (single-level wildcard), `#` (multi-level wildcard)

#### Example:
```python
# Subscribe to all temperature topics
client.subscribe("home/+/temperature")
# Subscribe to all topics under home
client.subscribe("home/#")
```

### Retained Messages
A retained message is stored by the broker and sent to new subscribers.

#### Example:
```python
client.publish("home/status", "online", retain=True)
```

### Last Will and Testament
Set a message to be sent if the client disconnects unexpectedly.

#### Example:
```python
client.will_set("home/status", "offline", retain=True)
```

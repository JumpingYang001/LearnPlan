# Quality of Service and Reliability

## Explanation
This section covers the three QoS levels, delivery guarantees, session management, and reliable message delivery.

### QoS Levels
- 0: At most once (fire and forget)
- 1: At least once (acknowledged delivery)
- 2: Exactly once (assured delivery)

#### Example:
```python
# Publish with different QoS levels
client.publish("test/qos0", "msg", qos=0)
client.publish("test/qos1", "msg", qos=1)
client.publish("test/qos2", "msg", qos=2)
```

### Session Management
- Clean session vs. persistent session

#### Example:
```python
client = mqtt.Client(clean_session=False)
```

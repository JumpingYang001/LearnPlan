# MQTT 5.0 Features

## Explanation
This section covers new features in MQTT 5.0, such as message expiry, user properties, shared subscriptions, and topic aliases.

### Message Expiry
```python
client.publish("test/topic", "msg", properties={"message_expiry_interval": 60})
```

### User Properties
```python
props = mqtt.Properties(mqtt.PacketTypes.PUBLISH)
props.UserProperty = ("key", "value")
client.publish("test/topic", "msg", properties=props)
```

### Shared Subscriptions
- Format: `$share/group/topic`

### Topic Aliases
- Reduce packet size by using aliases

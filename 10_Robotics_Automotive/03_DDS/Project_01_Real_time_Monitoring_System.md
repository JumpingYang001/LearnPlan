# Project: Real-time Monitoring System

## Description
Build a distributed monitoring system using DDS. Implement different QoS profiles for various data types and create dashboards for real-time data visualization.

## Example Code (Pseudocode)
```cpp
// Publisher for sensor data
DDSDomainParticipant participant = create_participant();
DDSPublisher publisher = participant.create_publisher();
DDSTopic topic = participant.create_topic("SensorData", SensorType);
DDSDataWriter writer = publisher.create_datawriter(topic);
SensorType data;
data.value = get_sensor_value();
writer.write(data);

// Subscriber for dashboard
DDSSubscriber subscriber = participant.create_subscriber();
DDSDataReader reader = subscriber.create_datareader(topic);
reader.set_listener(update_dashboard);
```

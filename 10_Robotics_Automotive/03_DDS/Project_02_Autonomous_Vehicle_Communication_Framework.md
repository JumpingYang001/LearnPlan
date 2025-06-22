# Project: Autonomous Vehicle Communication Framework

## Description
Implement inter-component communication for autonomous vehicles using DDS. Create data models for sensor fusion and ensure reliability and timing guarantees with QoS.

## Example Code (Pseudocode)
```cpp
// Define data model for sensor fusion
struct SensorFusionData {
  float lidar;
  float radar;
  float camera;
};

// Publisher for sensor fusion
DDSDomainParticipant participant = create_participant();
DDSPublisher publisher = participant.create_publisher();
DDSTopic topic = participant.create_topic("SensorFusion", SensorFusionData);
DDSDataWriter writer = publisher.create_datawriter(topic);
SensorFusionData data = get_fused_data();
writer.write(data);
```

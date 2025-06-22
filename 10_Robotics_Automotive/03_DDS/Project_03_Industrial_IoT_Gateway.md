# Project: Industrial IoT Gateway

## Description
Develop a gateway connecting industrial equipment to DDS. Implement data transformation, filtering, and ensure security and reliability.

## Example Code (Pseudocode)
```cpp
// Gateway reads from equipment and publishes to DDS
EquipmentData raw = read_equipment();
DDSDomainParticipant participant = create_participant();
DDSPublisher publisher = participant.create_publisher();
DDSTopic topic = participant.create_topic("EquipmentData", EquipmentType);
DDSDataWriter writer = publisher.create_datawriter(topic);
EquipmentType transformed = transform_data(raw);
writer.write(transformed);
```

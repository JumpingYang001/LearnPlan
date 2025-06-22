# Project: Multi-robot Coordination System

## Description
Implement communication between multiple robots using DDS. Create shared world models and develop coordination and task allocation algorithms.

## Example Code (Pseudocode)
```cpp
// Robot publishes its state
DDSDomainParticipant participant = create_participant();
DDSPublisher publisher = participant.create_publisher();
DDSTopic topic = participant.create_topic("RobotState", StateType);
DDSDataWriter writer = publisher.create_datawriter(topic);
StateType state = get_robot_state();
writer.write(state);

// Robots subscribe to shared world model
DDSSubscriber subscriber = participant.create_subscriber();
DDSDataReader reader = subscriber.create_datareader(topic);
reader.set_listener(update_shared_model);
```

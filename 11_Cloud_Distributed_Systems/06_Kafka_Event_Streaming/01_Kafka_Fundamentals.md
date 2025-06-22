# Kafka Fundamentals

## Description
Covers Kafka's architecture, components, topics, partitions, offsets, producers, consumers, and message delivery semantics.

## Example Code
```shell
# Start a Kafka broker (example)
$KAFKA_HOME/bin/kafka-server-start.sh $KAFKA_HOME/config/server.properties

# Create a topic
$KAFKA_HOME/bin/kafka-topics.sh --create --topic test --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

# Produce a message
$KAFKA_HOME/bin/kafka-console-producer.sh --topic test --bootstrap-server localhost:9092

# Consume a message
$KAFKA_HOME/bin/kafka-console-consumer.sh --topic test --from-beginning --bootstrap-server localhost:9092
```

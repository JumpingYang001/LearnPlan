# Kafka Cluster Setup and Administration

## Description
Master Kafka cluster setup, configuration, ZooKeeper's role, broker tuning, monitoring, and management.

## Example Code
```shell
# Start ZooKeeper
$KAFKA_HOME/bin/zookeeper-server-start.sh $KAFKA_HOME/config/zookeeper.properties

# Start Kafka broker
$KAFKA_HOME/bin/kafka-server-start.sh $KAFKA_HOME/config/server.properties

# Broker configuration example (server.properties)
broker.id=0
log.dirs=/tmp/kafka-logs
zookeeper.connect=localhost:2181
```

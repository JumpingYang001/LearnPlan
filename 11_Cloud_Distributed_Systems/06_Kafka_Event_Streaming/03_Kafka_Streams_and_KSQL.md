# Kafka Streams and KSQL

## Description
Stream processing concepts, Kafka Streams API, KSQL for SQL-like stream processing, and application examples.

## Example Code
```java
// Kafka Streams Java Example
StreamsBuilder builder = new StreamsBuilder();
KStream<String, String> source = builder.stream("input-topic");
KStream<String, String> uppercased = source.mapValues(String::toUpperCase);
uppercased.to("output-topic");

KafkaStreams streams = new KafkaStreams(builder.build(), config);
streams.start();
```

```sql
-- KSQL Example
CREATE STREAM input_stream (name VARCHAR) WITH (KAFKA_TOPIC='input-topic', VALUE_FORMAT='JSON');
CREATE STREAM upper_stream AS SELECT UCASE(name) AS name FROM input_stream;
```

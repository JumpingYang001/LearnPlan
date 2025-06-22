# Project: Data Integration Platform

## Description
Build a comprehensive data pipeline using Kafka Connect, implement CDC (Change Data Capture) from databases, and create transformations and routing of events.

## Example Code
```json
// Kafka Connect CDC Source Connector config
{
  "name": "cdc-source-connector",
  "config": {
    "connector.class": "io.debezium.connector.mysql.MySqlConnector",
    "database.hostname": "localhost",
    "database.port": "3306",
    "database.user": "debezium",
    "database.password": "dbz",
    "database.server.id": "184054",
    "database.server.name": "dbserver1",
    "database.include.list": "inventory",
    "database.history.kafka.bootstrap.servers": "localhost:9092",
    "database.history.kafka.topic": "schema-changes.inventory"
  }
}
```

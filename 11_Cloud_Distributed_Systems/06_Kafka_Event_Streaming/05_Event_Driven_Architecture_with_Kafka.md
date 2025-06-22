# Event-Driven Architecture with Kafka

## Description
Principles of event-driven architecture, event sourcing, CQRS, domain/integration events, and microservices implementation.

## Example Code
```java
// Event Sourcing Example (pseudo-code)
class OrderService {
  void createOrder(Order order) {
    // Save event to Kafka
    kafkaProducer.send("order-events", orderCreatedEvent);
  }
}
```

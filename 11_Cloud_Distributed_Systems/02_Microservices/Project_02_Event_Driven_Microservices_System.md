# Project 2: Event-Driven Microservices System

## Description
Develop a system using event sourcing and CQRS. Implement message brokers for communication and create event-driven workflows and processes.

## Example Code
```python
# Example: Publishing an event to RabbitMQ
import pika
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='events')
channel.basic_publish(exchange='', routing_key='events', body='OrderCreated')
connection.close()
```

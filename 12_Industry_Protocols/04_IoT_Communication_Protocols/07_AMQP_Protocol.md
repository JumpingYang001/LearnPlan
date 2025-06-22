# AMQP (Advanced Message Queuing Protocol)

## Explanation
AMQP is a protocol for message-oriented middleware, supporting reliable queuing, routing, and transactions. Used in IoT for robust messaging between devices and cloud.

## Example
```python
# Example: AMQP Publisher (pika)
import pika
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='iot')
channel.basic_publish(exchange='', routing_key='iot', body='Hello IoT!')
connection.close()
```

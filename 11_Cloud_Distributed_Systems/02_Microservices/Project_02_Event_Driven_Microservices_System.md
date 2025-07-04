# Project 2: Event-Driven Microservices System

*Duration: 3-4 weeks | Difficulty: Advanced | Prerequisites: Microservices basics, Message queues, Docker*

## Project Overview

Build a comprehensive event-driven microservices system that demonstrates modern distributed architecture patterns. This project implements **Event Sourcing**, **CQRS (Command Query Responsibility Segregation)**, and **Saga Pattern** using multiple microservices that communicate through message brokers.

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │  Event Store    │    │  Message Broker │
│   (Kong/Nginx)  │    │  (EventStore)   │    │   (RabbitMQ)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Event-Driven Microservices                  │
├─────────────────┬─────────────────┬─────────────────┬───────────┤
│ Order Service   │ Payment Service │ Inventory Svc   │ Notify Svc│
│ (Command/Query) │ (Command/Query) │ (Command/Query) │ (Query)   │
├─────────────────┼─────────────────┼─────────────────┼───────────┤
│ PostgreSQL      │ PostgreSQL      │ MongoDB         │ Redis     │
│ (Write DB)      │ (Write DB)      │ (Write DB)      │ (Read DB) │
├─────────────────┼─────────────────┼─────────────────┼───────────┤
│ Elasticsearch   │ Elasticsearch   │ Elasticsearch   │           │
│ (Read DB)       │ (Read DB)       │ (Read DB)       │           │
└─────────────────┴─────────────────┴─────────────────┴───────────┘
```

### Core Concepts Implemented

#### 1. **Event Sourcing**
Store all changes as a sequence of events rather than current state.

#### 2. **CQRS (Command Query Responsibility Segregation)**
Separate read and write operations using different data models.

#### 3. **Saga Pattern**
Manage distributed transactions across multiple microservices.

#### 4. **Domain-Driven Design (DDD)**
Organize services around business domains.

## Technical Stack

### Backend Services
- **Languages**: Python (FastAPI), Node.js (Express), Go (Gin)
- **Databases**: PostgreSQL (write), Elasticsearch (read), MongoDB, Redis
- **Message Brokers**: RabbitMQ, Apache Kafka
- **Event Store**: EventStore DB or custom implementation

### Infrastructure
- **Containerization**: Docker, Docker Compose
- **Orchestration**: Kubernetes (optional)
- **Monitoring**: Prometheus, Grafana, Jaeger (tracing)
- **API Gateway**: Kong or Nginx

### Development Tools
- **Testing**: pytest, Jest, Go test
- **Documentation**: OpenAPI/Swagger
- **CI/CD**: GitHub Actions, GitLab CI

## Detailed Implementation

### Phase 1: Foundation Setup

#### 1.1 Project Structure
```
event-driven-microservices/
├── docker-compose.yml
├── infrastructure/
│   ├── rabbitmq/
│   ├── eventstore/
│   ├── databases/
│   └── monitoring/
├── services/
│   ├── order-service/
│   ├── payment-service/
│   ├── inventory-service/
│   ├── notification-service/
│   └── api-gateway/
├── shared/
│   ├── events/
│   ├── messaging/
│   └── common/
└── tests/
    ├── integration/
    └── e2e/
```

#### 1.2 Docker Compose Infrastructure
```yaml
# docker-compose.yml
version: '3.8'

services:
  # Message Broker
  rabbitmq:
    image: rabbitmq:3-management
    container_name: rabbitmq
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: password
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq

  # Event Store
  eventstore:
    image: eventstore/eventstore:22.10.0-buster-slim
    container_name: eventstore
    ports:
      - "2113:2113"
      - "1113:1113"
    environment:
      - EVENTSTORE_CLUSTER_SIZE=1
      - EVENTSTORE_RUN_PROJECTIONS=All
      - EVENTSTORE_START_STANDARD_PROJECTIONS=true
      - EVENTSTORE_EXT_TCP_PORT=1113
      - EVENTSTORE_HTTP_PORT=2113
      - EVENTSTORE_INSECURE=true
      - EVENTSTORE_ENABLE_EXTERNAL_TCP=true
      - EVENTSTORE_ENABLE_ATOM_PUB_OVER_HTTP=true
    volumes:
      - eventstore_data:/var/lib/eventstore

  # Write Database (PostgreSQL)
  postgres:
    image: postgres:15
    container_name: postgres
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: microservices
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # Read Database (Elasticsearch)
  elasticsearch:
    image: elasticsearch:8.8.0
    container_name: elasticsearch
    ports:
      - "9200:9200"
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  # Cache (Redis)
  redis:
    image: redis:7-alpine
    container_name: redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # MongoDB for Inventory Service
  mongodb:
    image: mongo:6
    container_name: mongodb
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: password
    volumes:
      - mongodb_data:/data/db

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./infrastructure/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  rabbitmq_data:
  eventstore_data:
  postgres_data:
  elasticsearch_data:
  redis_data:
  mongodb_data:
  grafana_data:
```

### Phase 2: Shared Components

#### 2.1 Event Definitions
```python
# shared/events/base_event.py
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any
import uuid
import json

class BaseEvent(ABC):
    def __init__(self, aggregate_id: str, event_data: Dict[str, Any]):
        self.event_id = str(uuid.uuid4())
        self.aggregate_id = aggregate_id
        self.event_type = self.__class__.__name__
        self.event_data = event_data
        self.timestamp = datetime.utcnow().isoformat()
        self.version = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'aggregate_id': self.aggregate_id,
            'event_type': self.event_type,
            'event_data': self.event_data,
            'timestamp': self.timestamp,
            'version': self.version
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        event = cls(data['aggregate_id'], data['event_data'])
        event.event_id = data['event_id']
        event.timestamp = data['timestamp']
        event.version = data['version']
        return event

# Domain Events
class OrderCreated(BaseEvent):
    def __init__(self, order_id: str, customer_id: str, items: list, total_amount: float):
        super().__init__(order_id, {
            'customer_id': customer_id,
            'items': items,
            'total_amount': total_amount,
            'status': 'created'
        })

class OrderConfirmed(BaseEvent):
    def __init__(self, order_id: str):
        super().__init__(order_id, {'status': 'confirmed'})

class OrderCancelled(BaseEvent):
    def __init__(self, order_id: str, reason: str):
        super().__init__(order_id, {'status': 'cancelled', 'reason': reason})

class PaymentProcessed(BaseEvent):
    def __init__(self, payment_id: str, order_id: str, amount: float, status: str):
        super().__init__(payment_id, {
            'order_id': order_id,
            'amount': amount,
            'status': status
        })

class InventoryReserved(BaseEvent):
    def __init__(self, reservation_id: str, order_id: str, items: list):
        super().__init__(reservation_id, {
            'order_id': order_id,
            'items': items,
            'status': 'reserved'
        })

class InventoryReleased(BaseEvent):
    def __init__(self, reservation_id: str, order_id: str, items: list):
        super().__init__(reservation_id, {
            'order_id': order_id,
            'items': items,
            'status': 'released'
        })
```

#### 2.2 Message Broker Abstraction
```python
# shared/messaging/message_broker.py
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any
import pika
import json
import logging

logger = logging.getLogger(__name__)

class MessageBroker(ABC):
    @abstractmethod
    def publish(self, exchange: str, routing_key: str, message: Dict[str, Any]):
        pass
    
    @abstractmethod
    def subscribe(self, queue: str, callback: Callable):
        pass
    
    @abstractmethod
    def close(self):
        pass

class RabbitMQBroker(MessageBroker):
    def __init__(self, host='localhost', port=5672, username='admin', password='password'):
        self.connection_params = pika.ConnectionParameters(
            host=host,
            port=port,
            credentials=pika.PlainCredentials(username, password)
        )
        self.connection = None
        self.channel = None
        self._connect()

    def _connect(self):
        try:
            self.connection = pika.BlockingConnection(self.connection_params)
            self.channel = self.connection.channel()
            
            # Declare exchanges
            self.channel.exchange_declare(exchange='events', exchange_type='topic', durable=True)
            self.channel.exchange_declare(exchange='commands', exchange_type='direct', durable=True)
            
            logger.info("Connected to RabbitMQ")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    def publish(self, exchange: str, routing_key: str, message: Dict[str, Any]):
        try:
            self.channel.basic_publish(
                exchange=exchange,
                routing_key=routing_key,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                    content_type='application/json'
                )
            )
            logger.info(f"Published message to {exchange}/{routing_key}")
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            self._reconnect()
            raise

    def subscribe(self, queue: str, callback: Callable):
        try:
            self.channel.queue_declare(queue=queue, durable=True)
            
            def wrapper(ch, method, properties, body):
                try:
                    message = json.loads(body)
                    callback(message)
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            
            self.channel.basic_consume(queue=queue, on_message_callback=wrapper)
            logger.info(f"Subscribed to queue: {queue}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to queue {queue}: {e}")
            raise

    def start_consuming(self):
        try:
            logger.info("Starting to consume messages")
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Stopping consumption")
            self.channel.stop_consuming()

    def _reconnect(self):
        try:
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            self._connect()
        except Exception as e:
            logger.error(f"Failed to reconnect: {e}")

    def close(self):
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logger.info("RabbitMQ connection closed")
```

#### 2.3 Event Store Implementation
```python
# shared/event_store/event_store.py
from abc import ABC, abstractmethod
from typing import List, Optional
import requests
import json
import uuid

class EventStore(ABC):
    @abstractmethod
    def append_events(self, stream_name: str, events: List[dict], expected_version: int = -1):
        pass
    
    @abstractmethod
    def get_events(self, stream_name: str, from_version: int = 0) -> List[dict]:
        pass

class EventStoreDBClient(EventStore):
    def __init__(self, host='localhost', port=2113):
        self.base_url = f"http://{host}:{port}"
        
    def append_events(self, stream_name: str, events: List[dict], expected_version: int = -1):
        url = f"{self.base_url}/streams/{stream_name}"
        headers = {
            'Content-Type': 'application/vnd.eventstore.events+json',
            'ES-ExpectedVersion': str(expected_version)
        }
        
        # Format events for EventStore
        formatted_events = []
        for event in events:
            formatted_events.append({
                'eventId': str(uuid.uuid4()),
                'eventType': event['event_type'],
                'data': event['event_data'],
                'metadata': {
                    'timestamp': event['timestamp'],
                    'aggregate_id': event['aggregate_id']
                }
            })
        
        response = requests.post(url, headers=headers, data=json.dumps(formatted_events))
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to append events: {response.text}")
            
        return response.status_code

    def get_events(self, stream_name: str, from_version: int = 0) -> List[dict]:
        url = f"{self.base_url}/streams/{stream_name}"
        headers = {'Accept': 'application/vnd.eventstore.atom+json'}
        
        response = requests.get(url, headers=headers)
        if response.status_code == 404:
            return []
        elif response.status_code != 200:
            raise Exception(f"Failed to get events: {response.text}")
            
        data = response.json()
        events = []
        
        for entry in data.get('entries', []):
            events.append({
                'event_id': entry['id'],
                'event_type': entry['eventType'],
                'event_data': entry['data'],
                'timestamp': entry['updated'],
                'stream_version': entry['eventNumber']
            })
            
        return events
```

### Phase 3: Order Service (CQRS Implementation)

#### 3.1 Order Service Structure
```
order-service/
├── app/
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── handlers.py
│   │   └── models.py
│   ├── queries/
│   │   ├── __init__.py
│   │   ├── handlers.py
│   │   └── models.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── aggregates.py
│   │   └── events.py
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── repositories.py
│   │   └── projections.py
│   └── main.py
├── requirements.txt
└── Dockerfile
```

#### 3.2 Domain Aggregate
```python
# order-service/app/domain/aggregates.py
from typing import List, Dict, Any
from enum import Enum
import uuid
from datetime import datetime

class OrderStatus(Enum):
    CREATED = "created"
    CONFIRMED = "confirmed"
    PAID = "paid"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class OrderItem:
    def __init__(self, product_id: str, quantity: int, price: float):
        self.product_id = product_id
        self.quantity = quantity
        self.price = price
        
    def total_price(self) -> float:
        return self.quantity * self.price

class OrderAggregate:
    def __init__(self, order_id: str = None):
        self.order_id = order_id or str(uuid.uuid4())
        self.customer_id = None
        self.items: List[OrderItem] = []
        self.status = OrderStatus.CREATED
        self.total_amount = 0.0
        self.created_at = None
        self.updated_at = None
        self._version = 0
        self._uncommitted_events = []

    def create_order(self, customer_id: str, items: List[Dict[str, Any]]) -> 'OrderCreated':
        if self.status != OrderStatus.CREATED:
            raise ValueError("Order can only be created once")
            
        self.customer_id = customer_id
        self.items = [OrderItem(item['product_id'], item['quantity'], item['price']) 
                     for item in items]
        self.total_amount = sum(item.total_price() for item in self.items)
        self.created_at = datetime.utcnow()
        self.updated_at = self.created_at
        
        event = OrderCreated(
            self.order_id,
            customer_id,
            [{'product_id': item.product_id, 'quantity': item.quantity, 'price': item.price} 
             for item in self.items],
            self.total_amount
        )
        
        self._apply_event(event)
        return event

    def confirm_order(self) -> 'OrderConfirmed':
        if self.status != OrderStatus.CREATED:
            raise ValueError(f"Cannot confirm order in status: {self.status}")
            
        self.status = OrderStatus.CONFIRMED
        self.updated_at = datetime.utcnow()
        
        event = OrderConfirmed(self.order_id)
        self._apply_event(event)
        return event

    def cancel_order(self, reason: str) -> 'OrderCancelled':
        if self.status in [OrderStatus.DELIVERED, OrderStatus.CANCELLED]:
            raise ValueError(f"Cannot cancel order in status: {self.status}")
            
        self.status = OrderStatus.CANCELLED
        self.updated_at = datetime.utcnow()
        
        event = OrderCancelled(self.order_id, reason)
        self._apply_event(event)
        return event

    def mark_as_paid(self) -> 'OrderPaid':
        if self.status != OrderStatus.CONFIRMED:
            raise ValueError(f"Cannot mark as paid order in status: {self.status}")
            
        self.status = OrderStatus.PAID
        self.updated_at = datetime.utcnow()
        
        event = OrderPaid(self.order_id)
        self._apply_event(event)
        return event

    def _apply_event(self, event):
        self._uncommitted_events.append(event)
        self._version += 1

    def get_uncommitted_events(self):
        return self._uncommitted_events.copy()

    def mark_events_as_committed(self):
        self._uncommitted_events.clear()

    @classmethod
    def from_events(cls, events: List[Any]) -> 'OrderAggregate':
        aggregate = cls()
        
        for event in events:
            if isinstance(event, OrderCreated):
                aggregate.customer_id = event.event_data['customer_id']
                aggregate.items = [OrderItem(item['product_id'], item['quantity'], item['price']) 
                                 for item in event.event_data['items']]
                aggregate.total_amount = event.event_data['total_amount']
                aggregate.status = OrderStatus.CREATED
                
            elif isinstance(event, OrderConfirmed):
                aggregate.status = OrderStatus.CONFIRMED
                
            elif isinstance(event, OrderCancelled):
                aggregate.status = OrderStatus.CANCELLED
                
            elif isinstance(event, OrderPaid):
                aggregate.status = OrderStatus.PAID
                
            aggregate._version += 1
            
        return aggregate
```

#### 3.3 Command Handlers
```python
# order-service/app/commands/handlers.py
from typing import Dict, Any
from app.domain.aggregates import OrderAggregate
from app.infrastructure.repositories import OrderRepository
from shared.messaging.message_broker import RabbitMQBroker
from shared.event_store.event_store import EventStoreDBClient
import logging

logger = logging.getLogger(__name__)

class OrderCommandHandler:
    def __init__(self):
        self.repository = OrderRepository(EventStoreDBClient())
        self.message_broker = RabbitMQBroker()

    async def handle_create_order(self, command: Dict[str, Any]) -> str:
        """Handle CreateOrder command"""
        try:
            # Create new order aggregate
            order = OrderAggregate()
            
            # Execute business logic
            event = order.create_order(
                command['customer_id'],
                command['items']
            )
            
            # Save to event store
            await self.repository.save(order)
            
            # Publish event
            self.message_broker.publish(
                exchange='events',
                routing_key='order.created',
                message=event.to_dict()
            )
            
            logger.info(f"Order created: {order.order_id}")
            return order.order_id
            
        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            raise

    async def handle_confirm_order(self, command: Dict[str, Any]) -> None:
        """Handle ConfirmOrder command"""
        try:
            # Load order from event store
            order = await self.repository.get_by_id(command['order_id'])
            
            if not order:
                raise ValueError(f"Order not found: {command['order_id']}")
            
            # Execute business logic
            event = order.confirm_order()
            
            # Save to event store
            await self.repository.save(order)
            
            # Publish event
            self.message_broker.publish(
                exchange='events',
                routing_key='order.confirmed',
                message=event.to_dict()
            )
            
            logger.info(f"Order confirmed: {order.order_id}")
            
        except Exception as e:
            logger.error(f"Failed to confirm order: {e}")
            raise

    async def handle_cancel_order(self, command: Dict[str, Any]) -> None:
        """Handle CancelOrder command"""
        try:
            order = await self.repository.get_by_id(command['order_id'])
            
            if not order:
                raise ValueError(f"Order not found: {command['order_id']}")
            
            event = order.cancel_order(command.get('reason', 'User requested'))
            
            await self.repository.save(order)
            
            self.message_broker.publish(
                exchange='events',
                routing_key='order.cancelled',
                message=event.to_dict()
            )
            
            logger.info(f"Order cancelled: {order.order_id}")
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            raise
```

#### 3.4 Query Handlers
```python
# order-service/app/queries/handlers.py
from typing import List, Dict, Any, Optional
from app.infrastructure.projections import OrderProjectionRepository
import logging

logger = logging.getLogger(__name__)

class OrderQueryHandler:
    def __init__(self):
        self.projection_repo = OrderProjectionRepository()

    async def get_order_by_id(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order details by ID"""
        try:
            order = await self.projection_repo.get_by_id(order_id)
            return order
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            raise

    async def get_orders_by_customer(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get all orders for a customer"""
        try:
            orders = await self.projection_repo.get_by_customer_id(customer_id)
            return orders
        except Exception as e:
            logger.error(f"Failed to get orders for customer {customer_id}: {e}")
            raise

    async def get_orders_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get all orders with specific status"""
        try:
            orders = await self.projection_repo.get_by_status(status)
            return orders
        except Exception as e:
            logger.error(f"Failed to get orders with status {status}: {e}")
            raise

    async def search_orders(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search orders with filters"""
        try:
            orders = await self.projection_repo.search(filters)
            return orders
        except Exception as e:
            logger.error(f"Failed to search orders: {e}")
            raise
```

#### 3.5 Repository Implementation
```python
# order-service/app/infrastructure/repositories.py
from typing import Optional
from app.domain.aggregates import OrderAggregate
from shared.event_store.event_store import EventStore
from shared.events.base_event import OrderCreated, OrderConfirmed, OrderCancelled
import logging

logger = logging.getLogger(__name__)

class OrderRepository:
    def __init__(self, event_store: EventStore):
        self.event_store = event_store

    async def get_by_id(self, order_id: str) -> Optional[OrderAggregate]:
        """Load order aggregate from event store"""
        try:
            stream_name = f"order-{order_id}"
            events_data = self.event_store.get_events(stream_name)
            
            if not events_data:
                return None
            
            # Convert event data to event objects
            events = []
            for event_data in events_data:
                event_type = event_data['event_type']
                
                if event_type == 'OrderCreated':
                    event = OrderCreated.from_dict(event_data)
                elif event_type == 'OrderConfirmed':
                    event = OrderConfirmed.from_dict(event_data)
                elif event_type == 'OrderCancelled':
                    event = OrderCancelled.from_dict(event_data)
                else:
                    logger.warning(f"Unknown event type: {event_type}")
                    continue
                    
                events.append(event)
            
            # Reconstruct aggregate from events
            order = OrderAggregate.from_events(events)
            order.order_id = order_id
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to load order {order_id}: {e}")
            raise

    async def save(self, order: OrderAggregate) -> None:
        """Save order aggregate events to event store"""
        try:
            uncommitted_events = order.get_uncommitted_events()
            
            if not uncommitted_events:
                return
            
            stream_name = f"order-{order.order_id}"
            events_data = [event.to_dict() for event in uncommitted_events]
            
            # Append events to stream
            self.event_store.append_events(
                stream_name,
                events_data,
                expected_version=order._version - len(uncommitted_events) - 1
            )
            
            # Mark events as committed
            order.mark_events_as_committed()
            
            logger.info(f"Saved {len(events_data)} events for order {order.order_id}")
            
        except Exception as e:
            logger.error(f"Failed to save order {order.order_id}: {e}")
            raise
```

#### 3.6 FastAPI Application
```python
# order-service/app/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.commands.handlers import OrderCommandHandler
from app.queries.handlers import OrderQueryHandler
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Order Service", version="1.0.0")

# Initialize handlers
command_handler = OrderCommandHandler()
query_handler = OrderQueryHandler()

# Pydantic models for API
class OrderItem(BaseModel):
    product_id: str
    quantity: int
    price: float

class CreateOrderRequest(BaseModel):
    customer_id: str
    items: List[OrderItem]

class ConfirmOrderRequest(BaseModel):
    order_id: str

class CancelOrderRequest(BaseModel):
    order_id: str
    reason: Optional[str] = "User requested"

# Command endpoints (Write operations)
@app.post("/orders", response_model=dict)
async def create_order(request: CreateOrderRequest):
    """Create a new order"""
    try:
        command = {
            'customer_id': request.customer_id,
            'items': [item.dict() for item in request.items]
        }
        
        order_id = await command_handler.handle_create_order(command)
        
        return {
            'order_id': order_id,
            'status': 'created',
            'message': 'Order created successfully'
        }
        
    except Exception as e:
        logger.error(f"Error creating order: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/orders/confirm")
async def confirm_order(request: ConfirmOrderRequest):
    """Confirm an order"""
    try:
        command = {'order_id': request.order_id}
        await command_handler.handle_confirm_order(command)
        
        return {
            'order_id': request.order_id,
            'status': 'confirmed',
            'message': 'Order confirmed successfully'
        }
        
    except Exception as e:
        logger.error(f"Error confirming order: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/orders/cancel")
async def cancel_order(request: CancelOrderRequest):
    """Cancel an order"""
    try:
        command = {
            'order_id': request.order_id,
            'reason': request.reason
        }
        await command_handler.handle_cancel_order(command)
        
        return {
            'order_id': request.order_id,
            'status': 'cancelled',
            'message': 'Order cancelled successfully'
        }
        
    except Exception as e:
        logger.error(f"Error cancelling order: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Query endpoints (Read operations)
@app.get("/orders/{order_id}")
async def get_order(order_id: str):
    """Get order by ID"""
    try:
        order = await query_handler.get_order_by_id(order_id)
        
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
            
        return order
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/customers/{customer_id}/orders")
async def get_customer_orders(customer_id: str):
    """Get all orders for a customer"""
    try:
        orders = await query_handler.get_orders_by_customer(customer_id)
        return {'orders': orders}
        
    except Exception as e:
        logger.error(f"Error getting customer orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders")
async def search_orders(
    status: Optional[str] = None,
    customer_id: Optional[str] = None,
    limit: int = 10,
    offset: int = 0
):
    """Search orders with filters"""
    try:
        filters = {
            'limit': limit,
            'offset': offset
        }
        
        if status:
            filters['status'] = status
        if customer_id:
            filters['customer_id'] = customer_id
            
        orders = await query_handler.search_orders(filters)
        return {'orders': orders}
        
    except Exception as e:
        logger.error(f"Error searching orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "order-service"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Phase 4: Payment Service

#### 4.1 Payment Service Implementation
```python
# payment-service/app/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
import logging
import random
import uuid
from shared.messaging.message_broker import RabbitMQBroker
from shared.events.base_event import PaymentProcessed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Payment Service", version="1.0.0")

# Initialize message broker
message_broker = RabbitMQBroker()

class ProcessPaymentRequest(BaseModel):
    order_id: str
    amount: float
    payment_method: str
    customer_id: str

class PaymentService:
    def __init__(self):
        self.message_broker = RabbitMQBroker()
        
    async def process_payment(self, order_id: str, amount: float, payment_method: str) -> Dict[str, Any]:
        """Simulate payment processing"""
        payment_id = str(uuid.uuid4())
        
        # Simulate payment processing delay
        await asyncio.sleep(1)
        
        # Simulate random payment success/failure
        success_rate = 0.9  # 90% success rate
        is_success = random.random() < success_rate
        
        status = "completed" if is_success else "failed"
        
        # Create and publish payment event
        event = PaymentProcessed(payment_id, order_id, amount, status)
        
        self.message_broker.publish(
            exchange='events',
            routing_key='payment.processed',
            message=event.to_dict()
        )
        
        logger.info(f"Payment {status} for order {order_id}: {payment_id}")
        
        return {
            'payment_id': payment_id,
            'order_id': order_id,
            'amount': amount,
            'status': status,
            'payment_method': payment_method
        }

payment_service = PaymentService()

@app.post("/payments/process")
async def process_payment(request: ProcessPaymentRequest):
    """Process payment for an order"""
    try:
        result = await payment_service.process_payment(
            request.order_id,
            request.amount,
            request.payment_method
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing payment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "payment-service"}

# Event handlers for saga orchestration
def handle_order_confirmed(message: Dict[str, Any]):
    """Handle OrderConfirmed event - initiate payment"""
    try:
        order_id = message['aggregate_id']
        amount = message['event_data'].get('total_amount', 0)
        
        logger.info(f"Order confirmed, initiating payment for order: {order_id}")
        
        # In a real system, you might:
        # 1. Validate the order
        # 2. Check customer payment methods
        # 3. Process payment asynchronously
        
        # For demo, we'll simulate automatic payment processing
        async def process_async():
            await payment_service.process_payment(order_id, amount, "credit_card")
            
        asyncio.create_task(process_async())
        
    except Exception as e:
        logger.error(f"Error handling order confirmed: {e}")

# Subscribe to events
def setup_event_subscriptions():
    """Setup event subscriptions for saga orchestration"""
    try:
        message_broker.subscribe('order.confirmed', handle_order_confirmed)
        logger.info("Payment service subscribed to events")
    except Exception as e:
        logger.error(f"Error setting up subscriptions: {e}")

@app.on_event("startup")
async def startup_event():
    """Setup event subscriptions on startup"""
    setup_event_subscriptions()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
```

### Phase 5: Inventory Service

#### 5.1 Inventory Service with MongoDB
```python
# inventory-service/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Dict, Any, Optional
import logging
import uuid
from datetime import datetime, timedelta
from shared.messaging.message_broker import RabbitMQBroker
from shared.events.base_event import InventoryReserved, InventoryReleased

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Inventory Service", version="1.0.0")

# MongoDB setup
MONGODB_URL = "mongodb://admin:password@localhost:27017"
client = AsyncIOMotorClient(MONGODB_URL)
database = client.inventory_db
products_collection = database.products
reservations_collection = database.reservations

# Message broker
message_broker = RabbitMQBroker()

class ProductModel(BaseModel):
    product_id: str
    name: str
    description: str
    price: float
    quantity_available: int
    reserved_quantity: int = 0

class ReserveItemsRequest(BaseModel):
    order_id: str
    items: List[Dict[str, Any]]  # [{"product_id": "...", "quantity": 5}]

class InventoryService:
    def __init__(self):
        self.message_broker = RabbitMQBroker()

    async def reserve_items(self, order_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reserve inventory items for an order"""
        reservation_id = str(uuid.uuid4())
        
        try:
            # Start transaction
            async with await client.start_session() as session:
                async with session.start_transaction():
                    reserved_items = []
                    
                    for item in items:
                        product_id = item['product_id']
                        quantity = item['quantity']
                        
                        # Check if product exists and has sufficient quantity
                        product = await products_collection.find_one(
                            {"product_id": product_id}, session=session
                        )
                        
                        if not product:
                            raise ValueError(f"Product not found: {product_id}")
                        
                        available = product['quantity_available'] - product.get('reserved_quantity', 0)
                        if available < quantity:
                            raise ValueError(f"Insufficient inventory for product {product_id}. Available: {available}, Requested: {quantity}")
                        
                        # Reserve the quantity
                        await products_collection.update_one(
                            {"product_id": product_id},
                            {"$inc": {"reserved_quantity": quantity}},
                            session=session
                        )
                        
                        reserved_items.append({
                            "product_id": product_id,
                            "quantity": quantity,
                            "price": product['price']
                        })
                    
                    # Create reservation record
                    reservation = {
                        "reservation_id": reservation_id,
                        "order_id": order_id,
                        "items": reserved_items,
                        "status": "reserved",
                        "created_at": datetime.utcnow(),
                        "expires_at": datetime.utcnow() + timedelta(minutes=30)  # 30 min reservation
                    }
                    
                    await reservations_collection.insert_one(reservation, session=session)
            
            # Publish inventory reserved event
            event = InventoryReserved(reservation_id, order_id, reserved_items)
            self.message_broker.publish(
                exchange='events',
                routing_key='inventory.reserved',
                message=event.to_dict()
            )
            
            logger.info(f"Reserved inventory for order {order_id}: {reservation_id}")
            
            return {
                "reservation_id": reservation_id,
                "order_id": order_id,
                "items": reserved_items,
                "status": "reserved"
            }
            
        except Exception as e:
            logger.error(f"Failed to reserve inventory for order {order_id}: {e}")
            raise

    async def release_reservation(self, reservation_id: str) -> None:
        """Release a reservation"""
        try:
            # Find reservation
            reservation = await reservations_collection.find_one(
                {"reservation_id": reservation_id}
            )
            
            if not reservation:
                raise ValueError(f"Reservation not found: {reservation_id}")
            
            if reservation['status'] != 'reserved':
                logger.warning(f"Reservation already {reservation['status']}: {reservation_id}")
                return
            
            # Start transaction
            async with await client.start_session() as session:
                async with session.start_transaction():
                    # Release reserved quantities
                    for item in reservation['items']:
                        await products_collection.update_one(
                            {"product_id": item['product_id']},
                            {"$inc": {"reserved_quantity": -item['quantity']}},
                            session=session
                        )
                    
                    # Update reservation status
                    await reservations_collection.update_one(
                        {"reservation_id": reservation_id},
                        {
                            "$set": {
                                "status": "released",
                                "released_at": datetime.utcnow()
                            }
                        },
                        session=session
                    )
            
            # Publish inventory released event
            event = InventoryReleased(reservation_id, reservation['order_id'], reservation['items'])
            self.message_broker.publish(
                exchange='events',
                routing_key='inventory.released',
                message=event.to_dict()
            )
            
            logger.info(f"Released reservation: {reservation_id}")
            
        except Exception as e:
            logger.error(f"Failed to release reservation {reservation_id}: {e}")
            raise

    async def confirm_reservation(self, reservation_id: str) -> None:
        """Confirm a reservation (convert to actual allocation)"""
        try:
            reservation = await reservations_collection.find_one(
                {"reservation_id": reservation_id}
            )
            
            if not reservation:
                raise ValueError(f"Reservation not found: {reservation_id}")
            
            if reservation['status'] != 'reserved':
                raise ValueError(f"Cannot confirm reservation in status: {reservation['status']}")
            
            # Start transaction
            async with await client.start_session() as session:
                async with session.start_transaction():
                    # Convert reserved to actual allocation
                    for item in reservation['items']:
                        await products_collection.update_one(
                            {"product_id": item['product_id']},
                            {
                                "$inc": {
                                    "quantity_available": -item['quantity'],
                                    "reserved_quantity": -item['quantity']
                                }
                            },
                            session=session
                        )
                    
                    # Update reservation status
                    await reservations_collection.update_one(
                        {"reservation_id": reservation_id},
                        {
                            "$set": {
                                "status": "confirmed",
                                "confirmed_at": datetime.utcnow()
                            }
                        },
                        session=session
                    )
            
            logger.info(f"Confirmed reservation: {reservation_id}")
            
        except Exception as e:
            logger.error(f"Failed to confirm reservation {reservation_id}: {e}")
            raise

inventory_service = InventoryService()

# API Endpoints
@app.post("/inventory/reserve")
async def reserve_inventory(request: ReserveItemsRequest):
    """Reserve inventory items for an order"""
    try:
        result = await inventory_service.reserve_items(request.order_id, request.items)
        return result
    except Exception as e:
        logger.error(f"Error reserving inventory: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/inventory/reservations/{reservation_id}/release")
async def release_reservation(reservation_id: str):
    """Release an inventory reservation"""
    try:
        await inventory_service.release_reservation(reservation_id)
        return {"message": "Reservation released successfully"}
    except Exception as e:
        logger.error(f"Error releasing reservation: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/inventory/reservations/{reservation_id}/confirm")
async def confirm_reservation(reservation_id: str):
    """Confirm an inventory reservation"""
    try:
        await inventory_service.confirm_reservation(reservation_id)
        return {"message": "Reservation confirmed successfully"}
    except Exception as e:
        logger.error(f"Error confirming reservation: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/inventory/products/{product_id}")
async def get_product(product_id: str):
    """Get product details"""
    try:
        product = await products_collection.find_one({"product_id": product_id})
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Remove MongoDB ObjectId for JSON serialization
        product.pop('_id', None)
        return product
    except Exception as e:
        logger.error(f"Error getting product: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "inventory-service"}

# Event handlers
def handle_order_created(message: Dict[str, Any]):
    """Handle OrderCreated event - reserve inventory"""
    try:
        order_id = message['aggregate_id']
        items = message['event_data']['items']
        
        logger.info(f"Order created, reserving inventory for order: {order_id}")
        
        # Convert items format
        inventory_items = [
            {"product_id": item['product_id'], "quantity": item['quantity']}
            for item in items
        ]
        
        # Reserve inventory asynchronously
        async def reserve_async():
            await inventory_service.reserve_items(order_id, inventory_items)
            
        asyncio.create_task(reserve_async())
        
    except Exception as e:
        logger.error(f"Error handling order created: {e}")

def handle_payment_failed(message: Dict[str, Any]):
    """Handle PaymentFailed event - release inventory"""
    try:
        order_id = message['event_data']['order_id']
        logger.info(f"Payment failed, releasing inventory for order: {order_id}")
        
        # Find and release reservation
        async def release_async():
            reservation = await reservations_collection.find_one({"order_id": order_id})
            if reservation:
                await inventory_service.release_reservation(reservation['reservation_id'])
                
        asyncio.create_task(release_async())
        
    except Exception as e:
        logger.error(f"Error handling payment failed: {e}")

def handle_order_cancelled(message: Dict[str, Any]):
    """Handle OrderCancelled event - release inventory"""
    try:
        order_id = message['aggregate_id']
        logger.info(f"Order cancelled, releasing inventory for order: {order_id}")
        
        # Find and release reservation
        async def release_async():
            reservation = await reservations_collection.find_one({"order_id": order_id})
            if reservation:
                await inventory_service.release_reservation(reservation['reservation_id'])
                
        asyncio.create_task(release_async())
        
    except Exception as e:
        logger.error(f"Error handling order cancelled: {e}")

# Setup event subscriptions
def setup_event_subscriptions():
    """Setup event subscriptions"""
    try:
        message_broker.subscribe('order.created', handle_order_created)
        message_broker.subscribe('payment.failed', handle_payment_failed)
        message_broker.subscribe('order.cancelled', handle_order_cancelled)
        logger.info("Inventory service subscribed to events")
    except Exception as e:
        logger.error(f"Error setting up subscriptions: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize database and setup subscriptions"""
    # Create sample products
    await initialize_sample_data()
    setup_event_subscriptions()

async def initialize_sample_data():
    """Initialize sample product data"""
    sample_products = [
        {
            "product_id": "prod-001",
            "name": "Laptop",
            "description": "High-performance laptop",
            "price": 999.99,
            "quantity_available": 100,
            "reserved_quantity": 0
        },
        {
            "product_id": "prod-002",
            "name": "Mouse",
            "description": "Wireless mouse",
            "price": 29.99,
            "quantity_available": 500,
            "reserved_quantity": 0
        },
        {
            "product_id": "prod-003",
            "name": "Keyboard",
            "description": "Mechanical keyboard",
            "price": 149.99,
            "quantity_available": 200,
            "reserved_quantity": 0
        }
    ]
    
    for product in sample_products:
        existing = await products_collection.find_one({"product_id": product["product_id"]})
        if not existing:
            await products_collection.insert_one(product)
            logger.info(f"Created sample product: {product['product_id']}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
```

### Phase 6: Saga Orchestration

#### 6.1 Order Processing Saga
```python
# shared/sagas/order_saga.py
from enum import Enum
from typing import Dict, Any, List
import logging
import asyncio
from datetime import datetime, timedelta
from shared.messaging.message_broker import RabbitMQBroker

logger = logging.getLogger(__name__)

class SagaStatus(Enum):
    STARTED = "started"
    INVENTORY_RESERVED = "inventory_reserved"
    PAYMENT_PROCESSED = "payment_processed"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"

class OrderProcessingSaga:
    def __init__(self, saga_id: str, order_id: str):
        self.saga_id = saga_id
        self.order_id = order_id
        self.status = SagaStatus.STARTED
        self.steps_completed = []
        self.compensation_steps = []
        self.created_at = datetime.utcnow()
        self.updated_at = self.created_at
        self.timeout_at = self.created_at + timedelta(minutes=30)  # 30 min timeout
        
        self.message_broker = RabbitMQBroker()

    async def start(self, order_data: Dict[str, Any]):
        """Start the order processing saga"""
        try:
            logger.info(f"Starting order saga: {self.saga_id} for order: {self.order_id}")
            
            # Step 1: Reserve Inventory
            await self._reserve_inventory(order_data)
            
        except Exception as e:
            logger.error(f"Failed to start saga {self.saga_id}: {e}")
            await self._handle_failure()

    async def _reserve_inventory(self, order_data: Dict[str, Any]):
        """Step 1: Reserve inventory"""
        try:
            # Send command to inventory service
            inventory_command = {
                'saga_id': self.saga_id,
                'order_id': self.order_id,
                'items': order_data['items'],
                'command_type': 'reserve_inventory'
            }
            
            self.message_broker.publish(
                exchange='commands',
                routing_key='inventory.reserve',
                message=inventory_command
            )
            
            logger.info(f"Saga {self.saga_id}: Inventory reservation requested")
            
        except Exception as e:
            logger.error(f"Saga {self.saga_id}: Failed to reserve inventory: {e}")
            await self._handle_failure()

    async def handle_inventory_reserved(self, event_data: Dict[str, Any]):
        """Handle successful inventory reservation"""
        try:
            self.status = SagaStatus.INVENTORY_RESERVED
            self.steps_completed.append('inventory_reserved')
            self.updated_at = datetime.utcnow()
            
            reservation_id = event_data['aggregate_id']
            self.compensation_steps.append({
                'step': 'release_inventory',
                'reservation_id': reservation_id
            })
            
            logger.info(f"Saga {self.saga_id}: Inventory reserved, proceeding to payment")
            
            # Step 2: Process Payment
            await self._process_payment(event_data)
            
        except Exception as e:
            logger.error(f"Saga {self.saga_id}: Error handling inventory reserved: {e}")
            await self._handle_failure()

    async def _process_payment(self, inventory_data: Dict[str, Any]):
        """Step 2: Process payment"""
        try:
            # Calculate total amount from reserved items
            total_amount = sum(item['price'] * item['quantity'] 
                             for item in inventory_data['event_data']['items'])
            
            payment_command = {
                'saga_id': self.saga_id,
                'order_id': self.order_id,
                'amount': total_amount,
                'payment_method': 'credit_card',
                'command_type': 'process_payment'
            }
            
            self.message_broker.publish(
                exchange='commands',
                routing_key='payment.process',
                message=payment_command
            )
            
            logger.info(f"Saga {self.saga_id}: Payment processing requested")
            
        except Exception as e:
            logger.error(f"Saga {self.saga_id}: Failed to process payment: {e}")
            await self._handle_failure()

    async def handle_payment_processed(self, event_data: Dict[str, Any]):
        """Handle payment processing result"""
        try:
            payment_status = event_data['event_data']['status']
            
            if payment_status == 'completed':
                await self._complete_saga(event_data)
            else:
                logger.warning(f"Saga {self.saga_id}: Payment failed")
                await self._handle_failure()
                
        except Exception as e:
            logger.error(f"Saga {self.saga_id}: Error handling payment processed: {e}")
            await self._handle_failure()

    async def _complete_saga(self, payment_data: Dict[str, Any]):
        """Complete the saga successfully"""
        try:
            self.status = SagaStatus.PAYMENT_PROCESSED
            self.steps_completed.append('payment_processed')
            
            # Step 3: Confirm Order and Inventory
            await self._confirm_order()
            await self._confirm_inventory_reservation()
            
            self.status = SagaStatus.COMPLETED
            self.updated_at = datetime.utcnow()
            
            logger.info(f"Saga {self.saga_id}: Completed successfully")
            
            # Send completion notification
            self._send_completion_notification()
            
        except Exception as e:
            logger.error(f"Saga {self.saga_id}: Error completing saga: {e}")
            await self._handle_failure()

    async def _confirm_order(self):
        """Confirm the order"""
        confirm_command = {
            'saga_id': self.saga_id,
            'order_id': self.order_id,
            'command_type': 'confirm_order'
        }
        
        self.message_broker.publish(
            exchange='commands',
            routing_key='order.confirm',
            message=confirm_command
        )

    async def _confirm_inventory_reservation(self):
        """Confirm inventory reservation"""
        # Find the reservation ID from compensation steps
        reservation_id = None
        for step in self.compensation_steps:
            if step['step'] == 'release_inventory':
                reservation_id = step['reservation_id']
                break
        
        if reservation_id:
            confirm_command = {
                'saga_id': self.saga_id,
                'reservation_id': reservation_id,
                'command_type': 'confirm_reservation'
            }
            
            self.message_broker.publish(
                exchange='commands',
                routing_key='inventory.confirm',
                message=confirm_command
            )

    async def _handle_failure(self):
        """Handle saga failure - run compensation"""
        try:
            self.status = SagaStatus.COMPENSATING
            self.updated_at = datetime.utcnow()
            
            logger.info(f"Saga {self.saga_id}: Starting compensation")
            
            # Run compensation steps in reverse order
            for compensation_step in reversed(self.compensation_steps):
                await self._execute_compensation_step(compensation_step)
            
            # Cancel the order
            await self._cancel_order()
            
            self.status = SagaStatus.COMPENSATED
            self.updated_at = datetime.utcnow()
            
            logger.info(f"Saga {self.saga_id}: Compensation completed")
            
        except Exception as e:
            logger.error(f"Saga {self.saga_id}: Error during compensation: {e}")
            self.status = SagaStatus.FAILED

    async def _execute_compensation_step(self, step: Dict[str, Any]):
        """Execute a compensation step"""
        if step['step'] == 'release_inventory':
            release_command = {
                'saga_id': self.saga_id,
                'reservation_id': step['reservation_id'],
                'command_type': 'release_reservation'
            }
            
            self.message_broker.publish(
                exchange='commands',
                routing_key='inventory.release',
                message=release_command
            )
            
            logger.info(f"Saga {self.saga_id}: Released inventory reservation")

    async def _cancel_order(self):
        """Cancel the order"""
        cancel_command = {
            'saga_id': self.saga_id,
            'order_id': self.order_id,
            'reason': 'Saga compensation',
            'command_type': 'cancel_order'
        }
        
        self.message_broker.publish(
            exchange='commands',
            routing_key='order.cancel',
            message=cancel_command
        )

    def _send_completion_notification(self):
        """Send completion notification"""
        notification = {
            'saga_id': self.saga_id,
            'order_id': self.order_id,
            'status': self.status.value,
            'completed_at': self.updated_at.isoformat(),
            'message': 'Order processed successfully'
        }
        
        self.message_broker.publish(
            exchange='events',
            routing_key='saga.completed',
            message=notification
        )

    def is_expired(self) -> bool:
        """Check if saga has expired"""
        return datetime.utcnow() > self.timeout_at
```

#### 6.2 Saga Orchestrator Service
```python
# saga-orchestrator/app/main.py
from fastapi import FastAPI, HTTPException
import asyncio
import logging
from typing import Dict, Any
import uuid
from shared.messaging.message_broker import RabbitMQBroker
from shared.sagas.order_saga import OrderProcessingSaga, SagaStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Saga Orchestrator", version="1.0.0")

# In-memory saga storage (use Redis/database in production)
active_sagas: Dict[str, OrderProcessingSaga] = {}

message_broker = RabbitMQBroker()

class SagaOrchestrator:
    def __init__(self):
        self.message_broker = RabbitMQBroker()

    async def start_order_saga(self, order_data: Dict[str, Any]) -> str:
        """Start a new order processing saga"""
        saga_id = str(uuid.uuid4())
        order_id = order_data['aggregate_id']
        
        try:
            saga = OrderProcessingSaga(saga_id, order_id)
            active_sagas[saga_id] = saga
            
            await saga.start(order_data['event_data'])
            
            logger.info(f"Started order saga: {saga_id} for order: {order_id}")
            return saga_id
            
        except Exception as e:
            logger.error(f"Failed to start order saga: {e}")
            raise

    async def handle_inventory_reserved(self, event_data: Dict[str, Any]):
        """Handle inventory reserved event"""
        # Find the saga that initiated this reservation
        saga = self._find_saga_by_order_id(event_data['event_data']['order_id'])
        
        if saga and saga.status == SagaStatus.STARTED:
            await saga.handle_inventory_reserved(event_data)

    async def handle_payment_processed(self, event_data: Dict[str, Any]):
        """Handle payment processed event"""
        order_id = event_data['event_data']['order_id']
        saga = self._find_saga_by_order_id(order_id)
        
        if saga and saga.status == SagaStatus.INVENTORY_RESERVED:
            await saga.handle_payment_processed(event_data)

    def _find_saga_by_order_id(self, order_id: str) -> OrderProcessingSaga:
        """Find active saga by order ID"""
        for saga in active_sagas.values():
            if saga.order_id == order_id:
                return saga
        return None

    async def cleanup_expired_sagas(self):
        """Cleanup expired sagas"""
        expired_sagas = []
        
        for saga_id, saga in active_sagas.items():
            if saga.is_expired() and saga.status not in [SagaStatus.COMPLETED, SagaStatus.COMPENSATED]:
                expired_sagas.append(saga_id)
                await saga._handle_failure()
        
        for saga_id in expired_sagas:
            del active_sagas[saga_id]
            logger.info(f"Cleaned up expired saga: {saga_id}")

orchestrator = SagaOrchestrator()

# Event handlers
def handle_order_created(message: Dict[str, Any]):
    """Handle OrderCreated event"""
    try:
        logger.info(f"Order created, starting saga for order: {message['aggregate_id']}")
        asyncio.create_task(orchestrator.start_order_saga(message))
    except Exception as e:
        logger.error(f"Error handling order created: {e}")

def handle_inventory_reserved(message: Dict[str, Any]):
    """Handle InventoryReserved event"""
    try:
        asyncio.create_task(orchestrator.handle_inventory_reserved(message))
    except Exception as e:
        logger.error(f"Error handling inventory reserved: {e}")

def handle_payment_processed(message: Dict[str, Any]):
    """Handle PaymentProcessed event"""
    try:
        asyncio.create_task(orchestrator.handle_payment_processed(message))
    except Exception as e:
        logger.error(f"Error handling payment processed: {e}")

# API endpoints for monitoring
@app.get("/sagas")
async def get_active_sagas():
    """Get all active sagas"""
    sagas = {}
    for saga_id, saga in active_sagas.items():
        sagas[saga_id] = {
            'saga_id': saga.saga_id,
            'order_id': saga.order_id,
            'status': saga.status.value,
            'steps_completed': saga.steps_completed,
            'created_at': saga.created_at.isoformat(),
            'updated_at': saga.updated_at.isoformat()
        }
    return {'active_sagas': sagas}

@app.get("/sagas/{saga_id}")
async def get_saga(saga_id: str):
    """Get specific saga details"""
    saga = active_sagas.get(saga_id)
    if not saga:
        raise HTTPException(status_code=404, detail="Saga not found")
    
    return {
        'saga_id': saga.saga_id,
        'order_id': saga.order_id,
        'status': saga.status.value,
        'steps_completed': saga.steps_completed,
        'compensation_steps': saga.compensation_steps,
        'created_at': saga.created_at.isoformat(),
        'updated_at': saga.updated_at.isoformat(),
        'timeout_at': saga.timeout_at.isoformat()
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "saga-orchestrator"}

# Setup event subscriptions
def setup_event_subscriptions():
    """Setup event subscriptions"""
    try:
        message_broker.subscribe('order.created', handle_order_created)
        message_broker.subscribe('inventory.reserved', handle_inventory_reserved)
        message_broker.subscribe('payment.processed', handle_payment_processed)
        logger.info("Saga orchestrator subscribed to events")
    except Exception as e:
        logger.error(f"Error setting up subscriptions: {e}")

# Background task for cleanup
async def cleanup_task():
    """Background task to cleanup expired sagas"""
    while True:
        try:
            await orchestrator.cleanup_expired_sagas()
            await asyncio.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    """Setup event subscriptions and start background tasks"""
    setup_event_subscriptions()
    asyncio.create_task(cleanup_task())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
```

### Phase 7: Read Model Projections

#### 7.1 Elasticsearch Projections
```python
# shared/projections/elasticsearch_projections.py
from elasticsearch import AsyncElasticsearch
from typing import Dict, Any, List
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ElasticsearchProjection:
    def __init__(self, host='localhost', port=9200):
        self.es = AsyncElasticsearch([f"http://{host}:{port}"])
        
    async def create_index_if_not_exists(self, index_name: str, mapping: Dict[str, Any]):
        """Create index with mapping if it doesn't exist"""
        try:
            if not await self.es.indices.exists(index=index_name):
                await self.es.indices.create(index=index_name, body={"mappings": mapping})
                logger.info(f"Created Elasticsearch index: {index_name}")
        except Exception as e:
            logger.error(f"Error creating index {index_name}: {e}")

    async def index_document(self, index: str, doc_id: str, document: Dict[str, Any]):
        """Index a document"""
        try:
            await self.es.index(index=index, id=doc_id, body=document)
            logger.debug(f"Indexed document {doc_id} in {index}")
        except Exception as e:
            logger.error(f"Error indexing document {doc_id}: {e}")

    async def update_document(self, index: str, doc_id: str, updates: Dict[str, Any]):
        """Update a document"""
        try:
            await self.es.update(index=index, id=doc_id, body={"doc": updates})
            logger.debug(f"Updated document {doc_id} in {index}")
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")

    async def search(self, index: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search documents"""
        try:
            response = await self.es.search(index=index, body=query)
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Error searching in {index}: {e}")
            return []

class OrderProjectionHandler:
    def __init__(self):
        self.es_projection = ElasticsearchProjection()
        self.index_name = "orders"
        
    async def initialize(self):
        """Initialize the projection"""
        mapping = {
            "properties": {
                "order_id": {"type": "keyword"},
                "customer_id": {"type": "keyword"},
                "status": {"type": "keyword"},
                "total_amount": {"type": "float"},
                "items": {
                    "type": "nested",
                    "properties": {
                        "product_id": {"type": "keyword"},
                        "quantity": {"type": "integer"},
                        "price": {"type": "float"
                    }
                },
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"},
                "payment_status": {"type": "keyword"},
                "inventory_status": {"type": "keyword"}
            }
        }
        await self.es_projection.create_index_if_not_exists(self.index_name, mapping)

    async def handle_order_created(self, event: Dict[str, Any]):
        """Handle OrderCreated event"""
        order_doc = {
            "order_id": event["aggregate_id"],
            "customer_id": event["event_data"]["customer_id"],
            "status": event["event_data"]["status"],
            "total_amount": event["event_data"]["total_amount"],
            "items": event["event_data"]["items"],
            "created_at": event["timestamp"],
            "updated_at": event["timestamp"],
            "payment_status": "pending",
            "inventory_status": "pending"
        }
        
        await self.es_projection.index_document(
            self.index_name,
            event["aggregate_id"],
            order_doc
        )

    async def handle_order_confirmed(self, event: Dict[str, Any]):
        """Handle OrderConfirmed event"""
        updates = {
            "status": "confirmed",
            "updated_at": event["timestamp"]
        }
        
        await self.es_projection.update_document(
            self.index_name,
            event["aggregate_id"],
            updates
        )

    async def handle_order_cancelled(self, event: Dict[str, Any]):
        """Handle OrderCancelled event"""
        updates = {
            "status": "cancelled",
            "updated_at": event["timestamp"],
            "cancellation_reason": event["event_data"].get("reason", "")
        }
        
        await self.es_projection.update_document(
            self.index_name,
            event["aggregate_id"],
            updates
        )

    async def handle_payment_processed(self, event: Dict[str, Any]):
        """Handle PaymentProcessed event"""
        order_id = event["event_data"]["order_id"]
        payment_status = event["event_data"]["status"]
        
        updates = {
            "payment_status": payment_status,
            "updated_at": event["timestamp"]
        }
        
        if payment_status == "completed":
            updates["status"] = "paid"
        elif payment_status == "failed":
            updates["status"] = "payment_failed"
        
        await self.es_projection.update_document(
            self.index_name,
            order_id,
            updates
        )

    async def handle_inventory_reserved(self, event: Dict[str, Any]):
        """Handle InventoryReserved event"""
        order_id = event["event_data"]["order_id"]
        
        updates = {
            "inventory_status": "reserved",
            "updated_at": event["timestamp"]
        }
        
        await self.es_projection.update_document(
            self.index_name,
            order_id,
            updates
        )

    async def search_orders(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search orders with filters"""
        query = {"query": {"bool": {"must": []}}}
        
        if "customer_id" in filters:
            query["query"]["bool"]["must"].append(
                {"term": {"customer_id": filters["customer_id"]}}
            )
        
        if "status" in filters:
            query["query"]["bool"]["must"].append(
                {"term": {"status": filters["status"]}}
            )
        
        if "date_from" in filters:
            query["query"]["bool"]["must"].append(
                {"range": {"created_at": {"gte": filters["date_from"]}}}
            )
        
        # Add pagination
        query["from"] = filters.get("offset", 0)
        query["size"] = filters.get("limit", 10)
        
        # Add sorting
        query["sort"] = [{"created_at": {"order": "desc"}}]
        
        return await self.es_projection.search(self.index_name, query)
```

#### 7.2 Query Service
```python
# query-service/app/main.py
from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict, Any, Optional
import logging
from shared.projections.elasticsearch_projections import OrderProjectionHandler
from shared.messaging.message_broker import RabbitMQBroker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Query Service", version="1.0.0")

# Initialize projection handlers
order_projection = OrderProjectionHandler()
message_broker = RabbitMQBroker()

@app.get("/orders/{order_id}")
async def get_order(order_id: str):
    """Get order by ID"""
    try:
        orders = await order_projection.search_orders({"order_id": order_id})
        
        if not orders:
            raise HTTPException(status_code=404, detail="Order not found")
        
        return orders[0]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting order {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders")
async def search_orders(
    customer_id: Optional[str] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Search orders with filters"""
    try:
        filters = {
            "limit": limit,
            "offset": offset
        }
        
        if customer_id:
            filters["customer_id"] = customer_id
        if status:
            filters["status"] = status
        if date_from:
            filters["date_from"] = date_from
        if date_to:
            filters["date_to"] = date_to
        
        orders = await order_projection.search_orders(filters)
        
        return {
            "orders": orders,
            "limit": limit,
            "offset": offset,
            "total": len(orders)
        }
        
    except Exception as e:
        logger.error(f"Error searching orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders/analytics/status-distribution")
async def get_status_distribution():
    """Get order status distribution"""
    try:
        # Aggregation query for status distribution
        agg_query = {
            "size": 0,
            "aggs": {
                "status_distribution": {
                    "terms": {
                        "field": "status",
                        "size": 10
                    }
                }
            }
        }
        
        result = await order_projection.es_projection.es.search(
            index=order_projection.index_name,
            body=agg_query
        )
        
        distribution = []
        for bucket in result["aggregations"]["status_distribution"]["buckets"]:
            distribution.append({
                "status": bucket["key"],
                "count": bucket["doc_count"]
            })
        
        return {"status_distribution": distribution}
        
    except Exception as e:
        logger.error(f"Error getting status distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders/analytics/revenue-by-day")
async def get_daily_revenue():
    """Get daily revenue analytics"""
    try:
        agg_query = {
            "size": 0,
            "aggs": {
                "daily_revenue": {
                    "date_histogram": {
                        "field": "created_at",
                        "calendar_interval": "day"
                    },
                    "aggs": {
                        "total_revenue": {
                            "sum": {
                                "field": "total_amount"
                            }
                        }
                    }
                }
            }
        }
        
        result = await order_projection.es_projection.es.search(
            index=order_projection.index_name,
            body=agg_query
        )
        
        revenue_data = []
        for bucket in result["aggregations"]["daily_revenue"]["buckets"]:
            revenue_data.append({
                "date": bucket["key_as_string"],
                "revenue": bucket["total_revenue"]["value"]
            })
        
        return {"daily_revenue": revenue_data}
        
    except Exception as e:
        logger.error(f"Error getting daily revenue: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "query-service"}

# Event handlers for projection updates
def handle_order_created(message: Dict[str, Any]):
    """Handle OrderCreated event"""
    try:
        asyncio.create_task(order_projection.handle_order_created(message))
    except Exception as e:
        logger.error(f"Error handling order created: {e}")

def handle_order_confirmed(message: Dict[str, Any]):
    """Handle OrderConfirmed event"""
    try:
        asyncio.create_task(order_projection.handle_order_confirmed(message))
    except Exception as e:
        logger.error(f"Error handling order confirmed: {e}")

def handle_order_cancelled(message: Dict[str, Any]):
    """Handle OrderCancelled event"""
    try:
        asyncio.create_task(order_projection.handle_order_cancelled(message))
    except Exception as e:
        logger.error(f"Error handling order cancelled: {e}")

def handle_payment_processed(message: Dict[str, Any]):
    """Handle PaymentProcessed event"""
    try:
        asyncio.create_task(order_projection.handle_payment_processed(message))
    except Exception as e:
        logger.error(f"Error handling payment processed: {e}")

def handle_inventory_reserved(message: Dict[str, Any]):
    """Handle InventoryReserved event"""
    try:
        asyncio.create_task(order_projection.handle_inventory_reserved(message))
    except Exception as e:
        logger.error(f"Error handling inventory reserved: {e}")

# Setup event subscriptions
def setup_event_subscriptions():
    """Setup event subscriptions for projection updates"""
    try:
        message_broker.subscribe('order.created', handle_order_created)
        message_broker.subscribe('order.confirmed', handle_order_confirmed)
        message_broker.subscribe('order.cancelled', handle_order_cancelled)
        message_broker.subscribe('payment.processed', handle_payment_processed)
        message_broker.subscribe('inventory.reserved', handle_inventory_reserved)
        logger.info("Query service subscribed to events")
    except Exception as e:
        logger.error(f"Error setting up subscriptions: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize projections and setup subscriptions"""
    await order_projection.initialize()
    setup_event_subscriptions()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
```

### Phase 8: API Gateway

#### 8.1 API Gateway Implementation
```python
# api-gateway/app/main.py
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
import logging
from typing import Dict, Any, Optional
import asyncio
import time
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API Gateway", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Service registry
SERVICES = {
    "order": "http://localhost:8001",
    "payment": "http://localhost:8002",
    "inventory": "http://localhost:8003",
    "saga": "http://localhost:8004",
    "query": "http://localhost:8005"
}

# Rate limiting (simple in-memory implementation)
request_counts = {}

class APIGateway:
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.circuit_breakers = {}

    async def forward_request(
        self,
        service: str,
        path: str,
        method: str,
        headers: Dict[str, str] = None,
        params: Dict[str, Any] = None,
        json_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Forward request to appropriate microservice"""
        
        if service not in SERVICES:
            raise HTTPException(status_code=404, detail=f"Service {service} not found")
        
        url = f"{SERVICES[service]}{path}"
        
        try:
            # Add request tracing
            trace_id = self._generate_trace_id()
            if headers is None:
                headers = {}
            headers["X-Trace-ID"] = trace_id
            
            # Circuit breaker check
            if self._is_circuit_open(service):
                raise HTTPException(status_code=503, detail=f"Service {service} is unavailable")
            
            # Make request
            start_time = time.time()
            
            if method.upper() == "GET":
                response = await self.http_client.get(url, headers=headers, params=params)
            elif method.upper() == "POST":
                response = await self.http_client.post(url, headers=headers, json=json_data)
            elif method.upper() == "PUT":
                response = await self.http_client.put(url, headers=headers, json=json_data)
            elif method.upper() == "DELETE":
                response = await self.http_client.delete(url, headers=headers)
            else:
                raise HTTPException(status_code=405, detail="Method not allowed")
            
            # Record response time
            response_time = time.time() - start_time
            logger.info(f"Request to {service}{path} took {response_time:.3f}s")
            
            # Update circuit breaker
            self._record_success(service)
            
            # Return response
            if response.status_code >= 400:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            return response.json() if response.content else {}
            
        except httpx.TimeoutException:
            self._record_failure(service)
            raise HTTPException(status_code=504, detail=f"Request to {service} timed out")
        except httpx.ConnectError:
            self._record_failure(service)
            raise HTTPException(status_code=503, detail=f"Service {service} is unavailable")
        except Exception as e:
            self._record_failure(service)
            logger.error(f"Error forwarding request to {service}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _generate_trace_id(self) -> str:
        """Generate unique trace ID for request tracking"""
        return f"trace-{int(time.time() * 1000)}-{id(asyncio.current_task())}"

    def _is_circuit_open(self, service: str) -> bool:
        """Check if circuit breaker is open for service"""
        if service not in self.circuit_breakers:
            self.circuit_breakers[service] = {
                "failures": 0,
                "last_failure": None,
                "state": "closed"  # closed, open, half-open
            }
        
        breaker = self.circuit_breakers[service]
        
        if breaker["state"] == "open":
            # Check if timeout period has passed
            if time.time() - breaker["last_failure"] > 60:  # 1 minute timeout
                breaker["state"] = "half-open"
                return False
            return True
        
        return False

    def _record_success(self, service: str):
        """Record successful request"""
        if service in self.circuit_breakers:
            self.circuit_breakers[service]["failures"] = 0
            self.circuit_breakers[service]["state"] = "closed"

    def _record_failure(self, service: str):
        """Record failed request"""
        if service not in self.circuit_breakers:
            self.circuit_breakers[service] = {
                "failures": 0,
                "last_failure": None,
                "state": "closed"
            }
        
        breaker = self.circuit_breakers[service]
        breaker["failures"] += 1
        breaker["last_failure"] = time.time()
        
        # Open circuit if too many failures
        if breaker["failures"] >= 5:
            breaker["state"] = "open"
            logger.warning(f"Circuit breaker opened for service: {service}")

gateway = APIGateway()

# Authentication middleware
async def authenticate(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication middleware"""
    # In production, validate JWT token here
    token = credentials.credentials
    if not token or token == "invalid":
        raise HTTPException(status_code=401, detail="Invalid authentication")
    return {"user_id": "user123", "roles": ["customer"]}

# Rate limiting middleware
async def rate_limit(request: Request):
    """Simple rate limiting middleware"""
    client_ip = request.client.host
    current_time = int(time.time())
    
    if client_ip not in request_counts:
        request_counts[client_ip] = []
    
    # Remove old requests (older than 1 minute)
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip]
        if current_time - req_time < 60
    ]
    
    # Check rate limit (60 requests per minute)
    if len(request_counts[client_ip]) >= 60:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    request_counts[client_ip].append(current_time)

# Order service routes
@app.post("/api/v1/orders")
async def create_order(
    request: Request,
    user: Dict = Depends(authenticate),
    _: None = Depends(rate_limit)
):
    """Create a new order"""
    body = await request.json()
    body["customer_id"] = user["user_id"]  # Set customer from auth
    
    return await gateway.forward_request(
        service="order",
        path="/orders",
        method="POST",
        json_data=body
    )

@app.get("/api/v1/orders/{order_id}")
async def get_order(
    order_id: str,
    user: Dict = Depends(authenticate),
    _: None = Depends(rate_limit)
):
    """Get order by ID"""
    return await gateway.forward_request(
        service="query",
        path=f"/orders/{order_id}",
        method="GET"
    )

@app.get("/api/v1/orders")
async def search_orders(
    request: Request,
    user: Dict = Depends(authenticate),
    _: None = Depends(rate_limit)
):
    """Search orders"""
    params = dict(request.query_params)
    params["customer_id"] = user["user_id"]  # Filter by authenticated user
    
    return await gateway.forward_request(
        service="query",
        path="/orders",
        method="GET",
        params=params
    )

@app.put("/api/v1/orders/confirm")
async def confirm_order(
    request: Request,
    user: Dict = Depends(authenticate),
    _: None = Depends(rate_limit)
):
    """Confirm an order"""
    body = await request.json()
    
    return await gateway.forward_request(
        service="order",
        path="/orders/confirm",
        method="PUT",
        json_data=body
    )

@app.put("/api/v1/orders/cancel")
async def cancel_order(
    request: Request,
    user: Dict = Depends(authenticate),
    _: None = Depends(rate_limit)
):
    """Cancel an order"""
    body = await request.json()
    
    return await gateway.forward_request(
        service="order",
        path="/orders/cancel",
        method="PUT",
        json_data=body
    )

# Payment service routes
@app.post("/api/v1/payments/process")
async def process_payment(
    request: Request,
    user: Dict = Depends(authenticate),
    _: None = Depends(rate_limit)
):
    """Process payment"""
    body = await request.json()
    body["customer_id"] = user["user_id"]
    
    return await gateway.forward_request(
        service="payment",
        path="/payments/process",
        method="POST",
        json_data=body
    )

# Inventory service routes
@app.get("/api/v1/inventory/products/{product_id}")
async def get_product(
    product_id: str,
    _: None = Depends(rate_limit)
):
    """Get product details"""
    return await gateway.forward_request(
        service="inventory",
        path=f"/inventory/products/{product_id}",
        method="GET"
    )

# Analytics routes
@app.get("/api/v1/analytics/orders/status-distribution")
async def get_order_status_distribution(
    user: Dict = Depends(authenticate),
    _: None = Depends(rate_limit)
):
    """Get order status distribution"""
    # Only allow admin users
    if "admin" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return await gateway.forward_request(
        service="query",
        path="/orders/analytics/status-distribution",
        method="GET"
    )

@app.get("/api/v1/analytics/orders/revenue-by-day")
async def get_daily_revenue(
    user: Dict = Depends(authenticate),
    _: None = Depends(rate_limit)
):
    """Get daily revenue analytics"""
    # Only allow admin users
    if "admin" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return await gateway.forward_request(
        service="query",
        path="/orders/analytics/revenue-by-day",
        method="GET"
    )

# Saga monitoring routes (admin only)
@app.get("/api/v1/sagas")
async def get_active_sagas(
    user: Dict = Depends(authenticate),
    _: None = Depends(rate_limit)
):
    """Get active sagas"""
    if "admin" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return await gateway.forward_request(
        service="saga",
        path="/sagas",
        method="GET"
    )

@app.get("/api/v1/sagas/{saga_id}")
async def get_saga(
    saga_id: str,
    user: Dict = Depends(authenticate),
    _: None = Depends(rate_limit)
):
    """Get saga details"""
    if "admin" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return await gateway.forward_request(
        service="saga",
        path=f"/sagas/{saga_id}",
        method="GET"
    )

# Health check routes
@app.get("/api/v1/health")
async def health_check():
    """Gateway health check"""
    return {"status": "healthy", "service": "api-gateway"}

@app.get("/api/v1/health/services")
async def services_health():
    """Check health of all services"""
    health_status = {}
    
    for service_name, service_url in SERVICES.items():
        try:
            response = await gateway.http_client.get(f"{service_url}/health", timeout=5.0)
            health_status[service_name] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            health_status[service_name] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    return {"services": health_status}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Phase 9: Testing Strategy

#### 9.1 Unit Tests
```python
# tests/unit/test_order_aggregate.py
import pytest
from datetime import datetime
from app.domain.aggregates import OrderAggregate, OrderStatus
from shared.events.base_event import OrderCreated, OrderConfirmed, OrderCancelled

class TestOrderAggregate:
    def test_create_order_success(self):
        """Test successful order creation"""
        # Arrange
        order = OrderAggregate()
        customer_id = "cust-123"
        items = [
            {"product_id": "prod-001", "quantity": 2, "price": 999.99},
            {"product_id": "prod-002", "quantity": 1, "price": 29.99}
        ]
        
        # Act
        event = order.create_order(customer_id, items)
        
        # Assert
        assert order.customer_id == customer_id
        assert len(order.items) == 2
        assert order.total_amount == 2029.97
        assert order.status == OrderStatus.CREATED
        assert isinstance(event, OrderCreated)
        assert len(order.get_uncommitted_events()) == 1

    def test_create_order_twice_fails(self):
        """Test that creating order twice fails"""
        # Arrange
        order = OrderAggregate()
        customer_id = "cust-123"
        items = [{"product_id": "prod-001", "quantity": 1, "price": 999.99}]
        
        # Act
        order.create_order(customer_id, items)
        
        # Assert
        with pytest.raises(ValueError, match="Order can only be created once"):
            order.create_order(customer_id, items)

    def test_confirm_order_success(self):
        """Test successful order confirmation"""
        # Arrange
        order = OrderAggregate()
        order.create_order("cust-123", [{"product_id": "prod-001", "quantity": 1, "price": 999.99}])
        order.mark_events_as_committed()
        
        # Act
        event = order.confirm_order()
        
        # Assert
        assert order.status == OrderStatus.CONFIRMED
        assert isinstance(event, OrderConfirmed)

    def test_confirm_order_invalid_status_fails(self):
        """Test that confirming order in invalid status fails"""
        # Arrange
        order = OrderAggregate()
        order.create_order("cust-123", [{"product_id": "prod-001", "quantity": 1, "price": 999.99}])
        order.cancel_order("test reason")
        
        # Act & Assert
        with pytest.raises(ValueError, match="Cannot confirm order in status"):
            order.confirm_order()

    def test_cancel_order_success(self):
        """Test successful order cancellation"""
        # Arrange
        order = OrderAggregate()
        order.create_order("cust-123", [{"product_id": "prod-001", "quantity": 1, "price": 999.99}])
        reason = "Customer requested"
        
        # Act
        event = order.cancel_order(reason)
        
        # Assert
        assert order.status == OrderStatus.CANCELLED
        assert isinstance(event, OrderCancelled)
        assert event.event_data["reason"] == reason

    def test_aggregate_from_events(self):
        """Test aggregate reconstruction from events"""
        # Arrange
        order_id = "order-123"
        events = [
            OrderCreated(order_id, "cust-123", [{"product_id": "prod-001", "quantity": 1, "price": 999.99}], 999.99),
            OrderConfirmed(order_id)
        ]
        
        # Act
        order = OrderAggregate.from_events(events)
        order.order_id = order_id
        
        # Assert
        assert order.order_id == order_id
        assert order.customer_id == "cust-123"
        assert order.status == OrderStatus.CONFIRMED
        assert order.total_amount == 999.99
        assert order._version == 2
```

#### 9.2 Integration Tests
```python
# tests/integration/test_order_saga.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from shared.sagas.order_saga import OrderProcessingSaga, SagaStatus

class TestOrderProcessingSaga:
    @pytest.fixture
    def saga(self):
        return OrderProcessingSaga("saga-123", "order-123")

    @pytest.fixture
    def mock_message_broker(self, monkeypatch):
        mock_broker = Mock()
        mock_broker.publish = Mock()
        
        # Mock the message broker import
        monkeypatch.setattr(
            "shared.sagas.order_saga.RabbitMQBroker",
            lambda: mock_broker
        )
        return mock_broker

    @pytest.mark.asyncio
    async def test_saga_start_success(self, saga, mock_message_broker):
        """Test successful saga start"""
        # Arrange
        order_data = {
            "items": [{"product_id": "prod-001", "quantity": 1, "price": 999.99}]
        }
        
        # Act
        await saga.start(order_data)
        
        # Assert
        assert saga.status == SagaStatus.STARTED
        mock_message_broker.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_inventory_reserved_success(self, saga, mock_message_broker):
        """Test successful inventory reservation handling"""
        # Arrange
        saga.status = SagaStatus.STARTED
        event_data = {
            "aggregate_id": "reservation-123",
            "event_data": {
                "order_id": "order-123",
                "items": [{"product_id": "prod-001", "quantity": 1, "price": 999.99}]
            }
        }
        
        # Act
        await saga.handle_inventory_reserved(event_data)
        
        # Assert
        assert saga.status == SagaStatus.INVENTORY_RESERVED
        assert "inventory_reserved" in saga.steps_completed
        assert len(saga.compensation_steps) == 1
        mock_message_broker.publish.assert_called()

    @pytest.mark.asyncio
    async def test_handle_payment_processed_success(self, saga, mock_message_broker):
        """Test successful payment processing"""
        # Arrange
        saga.status = SagaStatus.INVENTORY_RESERVED
        event_data = {
            "event_data": {
                "order_id": "order-123",
                "status": "completed",
                "amount": 999.99
            }
        }
        
        # Act
        await saga.handle_payment_processed(event_data)
        
        # Assert
        assert saga.status == SagaStatus.COMPLETED
        assert "payment_processed" in saga.steps_completed

    @pytest.mark.asyncio
    async def test_handle_payment_failed_triggers_compensation(self, saga, mock_message_broker):
        """Test payment failure triggers compensation"""
        # Arrange
        saga.status = SagaStatus.INVENTORY_RESERVED
        saga.compensation_steps = [{"step": "release_inventory", "reservation_id": "res-123"}]
        
        event_data = {
            "event_data": {
                "order_id": "order-123",
                "status": "failed",
                "amount": 999.99
            }
        }
        
        # Act
        await saga.handle_payment_processed(event_data)
        
        # Assert
        assert saga.status == SagaStatus.COMPENSATED
        # Verify compensation commands were sent
        calls = mock_message_broker.publish.call_args_list
        assert any("inventory.release" in str(call) for call in calls)
        assert any("order.cancel" in str(call) for call in calls)
```

#### 9.3 End-to-End Tests
```python
# tests/e2e/test_order_flow.py
import pytest
import httpx
import asyncio
import time
from typing import Dict, Any

@pytest.mark.asyncio
class TestOrderFlow:
    base_url = "http://localhost:8000/api/v1"
    
    @pytest.fixture
    def auth_headers(self):
        return {"Authorization": "Bearer valid-token"}

    async def test_complete_order_flow_success(self, auth_headers):
        """Test complete successful order flow"""
        async with httpx.AsyncClient() as client:
            # Step 1: Create order
            order_data = {
                "items": [
                    {"product_id": "prod-001", "quantity": 1, "price": 999.99},
                    {"product_id": "prod-002", "quantity": 2, "price": 29.99}
                ]
            }
            
            response = await client.post(
                f"{self.base_url}/orders",
                json=order_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            result = response.json()
            order_id = result["order_id"]
            assert result["status"] == "created"
            
            # Step 2: Wait for saga processing
            await asyncio.sleep(2)
            
            # Step 3: Check order status
            response = await client.get(
                f"{self.base_url}/orders/{order_id}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            order = response.json()
            
            # Order should be processed through the saga
            assert order["order_id"] == order_id
            assert order["status"] in ["confirmed", "paid", "created"]
            assert order["inventory_status"] in ["reserved", "pending"]
            
            # Step 4: Check inventory was reserved
            assert order["inventory_status"] == "reserved"
            
            # Step 5: Verify saga completed
            # (This would require admin access in real scenario)
            saga_response = await client.get(
                f"http://localhost:8004/sagas",  # Direct call to saga service
                headers=auth_headers
            )
            
            if saga_response.status_code == 200:
                sagas = saga_response.json()["active_sagas"]
                order_saga = next(
                    (saga for saga in sagas.values() if saga["order_id"] == order_id),
                    None
                )
                
                if order_saga:
                    assert order_saga["status"] in ["completed", "inventory_reserved", "payment_processed"]

    async def test_order_flow_with_inventory_failure(self, auth_headers):
        """Test order flow when inventory reservation fails"""
        async with httpx.AsyncClient() as client:
            # Step 1: Create order with invalid product
            order_data = {
                "items": [
                    {"product_id": "invalid-product", "quantity": 1000000, "price": 999.99}
                ]
            }
            
            response = await client.post(
                f"{self.base_url}/orders",
                json=order_data,
                headers=auth_headers
            )
            
            # Order creation should succeed initially
            assert response.status_code == 200
            result = response.json()
            order_id = result["order_id"]
            
            # Step 2: Wait for saga processing
            await asyncio.sleep(3)
            
            # Step 3: Check order was cancelled due to inventory failure
            response = await client.get(
                f"{self.base_url}/orders/{order_id}",
                headers=auth_headers
            )
            
            if response.status_code == 200:
                order = response.json()
                # Order should be cancelled due to saga compensation
                assert order["status"] in ["cancelled", "created"]

    async def test_order_search_and_analytics(self, auth_headers):
        """Test order search and analytics endpoints"""
        async with httpx.AsyncClient() as client:
            # Step 1: Search orders
            response = await client.get(
                f"{self.base_url}/orders",
                headers=auth_headers,
                params={"limit": 10, "offset": 0}
            )
            
            assert response.status_code == 200
            result = response.json()
            assert "orders" in result
            assert isinstance(result["orders"], list)
            
            # Step 2: Test analytics (requires admin token in real scenario)
            admin_headers = {"Authorization": "Bearer admin-token"}
            
            response = await client.get(
                f"{self.base_url}/analytics/orders/status-distribution",
                headers=admin_headers
            )
            
            # May return 403 for non-admin, which is expected
            assert response.status_code in [200, 403]
            
            if response.status_code == 200:
                result = response.json()
                assert "status_distribution" in result

    async def test_rate_limiting(self, auth_headers):
        """Test API gateway rate limiting"""
        async with httpx.AsyncClient() as client:
            # Make multiple rapid requests
            tasks = []
            for i in range(70):  # Exceed rate limit of 60/minute
                task = client.get(
                    f"{self.base_url}/health",
                    headers=auth_headers
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Some requests should be rate limited
            status_codes = []
            for response in responses:
                if isinstance(response, httpx.Response):
                    status_codes.append(response.status_code)
                elif isinstance(response, Exception):
                    # Handle connection errors as potential rate limiting
                    status_codes.append(429)
            
            # Should have some rate limiting responses
            assert 429 in status_codes or len([s for s in status_codes if s >= 400]) > 0

    async def test_circuit_breaker(self, auth_headers):
        """Test API gateway circuit breaker"""
        async with httpx.AsyncClient() as client:
            # This test would require stopping a service to trigger circuit breaker
            # For now, just test that the endpoint exists
            response = await client.get(
                f"{self.base_url}/health/services",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            result = response.json()
            assert "services" in result
            
            # Check that all expected services are monitored
            expected_services = ["order", "payment", "inventory", "saga", "query"]
            for service in expected_services:
                assert service in result["services"]
```

#### 9.4 Performance Tests
```python
# tests/performance/test_load.py
import asyncio
import httpx
import time
import statistics
from typing import List
import pytest

class TestPerformance:
    base_url = "http://localhost:8000/api/v1"
    
    async def test_order_creation_performance(self):
        """Test order creation performance under load"""
        auth_headers = {"Authorization": "Bearer valid-token"}
        
        async def create_order(session: httpx.AsyncClient, order_num: int):
            order_data = {
                "items": [
                    {"product_id": "prod-001", "quantity": 1, "price": 999.99}
                ]
            }
            
            start_time = time.time()
            try:
                response = await session.post(
                    f"{self.base_url}/orders",
                    json=order_data,
                    headers=auth_headers,
                    timeout=30.0
                )
                end_time = time.time()
                
                return {
                    "order_num": order_num,
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "success": response.status_code == 200
                }
            except Exception as e:
                end_time = time.time()
                return {
                    "order_num": order_num,
                    "status_code": 500,
                    "response_time": end_time - start_time,
                    "success": False,
                    "error": str(e)
                }
        
        # Create multiple concurrent orders
        num_orders = 50
        concurrency = 10
        
        async with httpx.AsyncClient() as client:
            semaphore = asyncio.Semaphore(concurrency)
            
            async def bounded_create_order(order_num: int):
                async with semaphore:
                    return await create_order(client, order_num)
            
            start_time = time.time()
            tasks = [bounded_create_order(i) for i in range(num_orders)]
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Analyze results
            successful_orders = [r for r in results if r["success"]]
            failed_orders = [r for r in results if not r["success"]]
            
            response_times = [r["response_time"] for r in successful_orders]
            
            total_time = end_time - start_time
            throughput = len(successful_orders) / total_time
            
            print(f"\nPerformance Test Results:")
            print(f"Total Orders: {num_orders}")
            print(f"Successful: {len(successful_orders)}")
            print(f"Failed: {len(failed_orders)}")
            print(f"Success Rate: {len(successful_orders)/num_orders*100:.1f}%")
            print(f"Total Time: {total_time:.2f}s")
            print(f"Throughput: {throughput:.2f} orders/sec")
            
            if response_times:
                print(f"Response Time Stats:")
                print(f"  Min: {min(response_times):.3f}s")
                print(f"  Max: {max(response_times):.3f}s")
                print(f"  Mean: {statistics.mean(response_times):.3f}s")
                print(f"  Median: {statistics.median(response_times):.3f}s")
                print(f"  95th percentile: {sorted(response_times)[int(len(response_times)*0.95)]:.3f}s")
            
            # Performance assertions
            assert len(successful_orders) >= num_orders * 0.95  # 95% success rate
            if response_times:
                assert statistics.mean(response_times) < 2.0  # Average response time < 2s
                assert max(response_times) < 10.0  # Max response time < 10s

    async def test_query_performance(self, auth_headers):
        """Test query performance"""
        auth_headers = {"Authorization": "Bearer valid-token"}
        
        async def query_orders(session: httpx.AsyncClient):
            start_time = time.time()
            try:
                response = await session.get(
                    f"{self.base_url}/orders",
                    headers=auth_headers,
                    params={"limit": 10},
                    timeout=10.0
                )
                end_time = time.time()
                
                return {
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "success": response.status_code == 200
                }
            except Exception as e:
                end_time = time.time()
                return {
                    "status_code": 500,
                    "response_time": end_time - start_time,
                    "success": False,
                    "error": str(e)
                }
        
        # Run concurrent queries
        num_queries = 100
        concurrency = 20
        
        async with httpx.AsyncClient() as client:
            semaphore = asyncio.Semaphore(concurrency)
            
            async def bounded_query():
                async with semaphore:
                    return await query_orders(client)
            
            start_time = time.time()
            tasks = [bounded_query() for _ in range(num_queries)]
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Analyze results
            successful_queries = [r for r in results if r["success"]]
            response_times = [r["response_time"] for r in successful_queries]
            
            total_time = end_time - start_time
            throughput = len(successful_queries) / total_time
            
            print(f"\nQuery Performance Results:")
            print(f"Total Queries: {num_queries}")
            print(f"Successful: {len(successful_queries)}")
            print(f"Success Rate: {len(successful_queries)/num_queries*100:.1f}%")
            print(f"Throughput: {throughput:.2f} queries/sec")
            
            if response_times:
                print(f"Query Response Time Stats:")
                print(f"  Mean: {statistics.mean(response_times):.3f}s")
                print(f"  95th percentile: {sorted(response_times)[int(len(response_times)*0.95)]:.3f}s")
            
            # Performance assertions for queries (should be faster than commands)
            assert len(successful_queries) >= num_queries * 0.98  # 98% success rate
            if response_times:
                assert statistics.mean(response_times) < 0.5  # Average query time < 500ms
```

### Phase 10: Deployment and Monitoring

#### 10.1 Kubernetes Deployment
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: event-microservices

---
# k8s/order-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
  namespace: event-microservices
spec:
  replicas: 3
  selector:
    matchLabels:
      app: order-service
  template:
    metadata:
      labels:
        app: order-service
    spec:
      containers:
      - name: order-service
        image: order-service:latest
        ports:
        - containerPort: 8001
        env:
        - name: DATABASE_URL
          value: "postgresql://postgres:password@postgres:5432/microservices"
        - name: RABBITMQ_URL
          value: "amqp://admin:password@rabbitmq:5672"
        - name: EVENTSTORE_URL
          value: "http://eventstore:2113"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"

---
apiVersion: v1
kind: Service
metadata:
  name: order-service
  namespace: event-microservices
spec:
  selector:
    app: order-service
  ports:
  - port: 8001
    targetPort: 8001
  type: ClusterIP

---
# k8s/api-gateway.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: event-microservices
spec:
  replicas: 2
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
      - name: api-gateway
        image: api-gateway:latest
        ports:
        - containerPort: 8000
        env:
        - name: ORDER_SERVICE_URL
          value: "http://order-service:8001"
        - name: PAYMENT_SERVICE_URL
          value: "http://payment-service:8002"
        - name: INVENTORY_SERVICE_URL
          value: "http://inventory-service:8003"
        - name: SAGA_SERVICE_URL
          value: "http://saga-orchestrator:8004"
        - name: QUERY_SERVICE_URL
          value: "http://query-service:8005"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: api-gateway
  namespace: event-microservices
spec:
  selector:
    app: api-gateway
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer

---
# k8s/monitoring.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: event-microservices
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus
      volumes:
      - name: prometheus-config
        configMap:
          name: prometheus-config

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: event-microservices
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'microservices'
      static_configs:
      - targets: 
        - 'order-service:8001'
        - 'payment-service:8002'
        - 'inventory-service:8003'
        - 'saga-orchestrator:8004'
        - 'query-service:8005'
        - 'api-gateway:8000'
      metrics_path: '/metrics'
```

#### 10.2 Monitoring and Observability
```python
# shared/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import functools
from typing import Callable
import logging

logger = logging.getLogger(__name__)

# Metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

active_connections = Gauge(
    'active_connections',
    'Active connections'
)

saga_count = Counter(
    'sagas_total',
    'Total sagas',
    ['type', 'status']
)

event_count = Counter(
    'events_total',
    'Total events',
    ['event_type', 'service']
)

message_queue_size = Gauge(
    'message_queue_size',
    'Message queue size',
    ['queue_name']
)

def metrics_middleware(request_handler: Callable):
    """Middleware to collect HTTP metrics"""
    @functools.wraps(request_handler)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            # Extract request info (this is FastAPI specific)
            request = args[0] if args else None
            method = request.method if hasattr(request, 'method') else 'UNKNOWN'
            path = request.url.path if hasattr(request, 'url') else 'UNKNOWN'
            
            # Execute request
            response = await request_handler(*args, **kwargs)
            
            # Record metrics
            status = getattr(response, 'status_code', 200)
            request_count.labels(method=method, endpoint=path, status=status).inc()
            request_duration.labels(method=method, endpoint=path).observe(time.time() - start_time)
            
            return response
            
        except Exception as e:
            # Record error
            request_count.labels(method=method, endpoint=path, status=500).inc()
            request_duration.labels(method=method, endpoint=path).observe(time.time() - start_time)
            raise
    
    return wrapper

class MetricsCollector:
    def __init__(self, service_name: str, port: int = 8000):
        self.service_name = service_name
        self.port = port
        
    def start_metrics_server(self):
        """Start Prometheus metrics server"""
        try:
            start_http_server(self.port)
            logger.info(f"Metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    def record_event(self, event_type: str):
        """Record event metric"""
        event_count.labels(event_type=event_type, service=self.service_name).inc()
    
    def record_saga(self, saga_type: str, status: str):
        """Record saga metric"""
        saga_count.labels(type=saga_type, status=status).inc()
    
    def update_queue_size(self, queue_name: str, size: int):
        """Update queue size metric"""
        message_queue_size.labels(queue_name=queue_name).set(size)
```

#### 10.3 Distributed Tracing
```python
# shared/tracing/tracer.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
import logging

logger = logging.getLogger(__name__)

class DistributedTracer:
    def __init__(self, service_name: str, jaeger_host: str = "localhost", jaeger_port: int = 14268):
        self.service_name = service_name
        self.jaeger_host = jaeger_host
        self.jaeger_port = jaeger_port
        self.tracer = None
        
    def initialize(self):
        """Initialize distributed tracing"""
        try:
            # Set up tracer provider
            trace.set_tracer_provider(
                TracerProvider(resource={"service.name": self.service_name})
            )
            
            # Set up Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.jaeger_host,
                agent_port=self.jaeger_port,
            )
            
            # Add span processor
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            # Get tracer
            self.tracer = trace.get_tracer(self.service_name)
            
            # Instrument FastAPI and HTTPX automatically
            FastAPIInstrumentor.instrument()
            HTTPXClientInstrumentor.instrument()
            
            logger.info(f"Distributed tracing initialized for {self.service_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}")
    
    def create_span(self, name: str, parent_context=None):
        """Create a new span"""
        if self.tracer:
            return self.tracer.start_span(name, context=parent_context)
        return None
    
    def add_event_to_span(self, span, event_name: str, attributes: dict = None):
        """Add event to span"""
        if span:
            span.add_event(event_name, attributes or {})
    
    def set_span_attribute(self, span, key: str, value):
        """Set span attribute"""
        if span:
            span.set_attribute(key, value)
```

## Learning Outcomes

Upon completing this project, you will have gained hands-on experience with:

### 🎯 **Core Concepts Mastered**
- **Event Sourcing**: Storing application state as a sequence of events
- **CQRS**: Separating read and write operations for optimal performance
- **Saga Pattern**: Managing distributed transactions across microservices
- **Domain-Driven Design**: Organizing services around business domains

### 🛠 **Technical Skills Developed**
- **Microservices Architecture**: Building scalable, distributed systems
- **Message-Driven Architecture**: Using RabbitMQ for async communication
- **Event Store Management**: Implementing event persistence and replay
- **API Gateway Patterns**: Authentication, rate limiting, circuit breakers
- **Distributed Monitoring**: Metrics, logging, and distributed tracing

### 📊 **DevOps and Operations**
- **Containerization**: Docker and Docker Compose orchestration
- **Kubernetes Deployment**: Production-ready container orchestration
- **Monitoring Stack**: Prometheus, Grafana, Jaeger integration
- **Testing Strategies**: Unit, integration, end-to-end, and performance testing

### 🔧 **Production Readiness**
- **Fault Tolerance**: Circuit breakers, timeouts, retries
- **Scalability**: Horizontal scaling patterns
- **Security**: Authentication, authorization, input validation
- **Performance**: Optimization techniques and load testing

## Next Steps

### 🚀 **Immediate Enhancements**
1. Add Redis caching layer for improved read performance
2. Implement event replay and disaster recovery
3. Add comprehensive integration with Kubernetes operators
4. Implement advanced saga patterns (choreography vs orchestration)

### 📈 **Advanced Features**
1. **Event Versioning**: Handle event schema evolution
2. **Snapshotting**: Optimize aggregate reconstruction
3. **Projection Rebuilding**: Automated read model regeneration
4. **Multi-tenant Architecture**: Isolate data per tenant

### 🔬 **Further Learning**
1. **Apache Kafka**: Alternative message broker implementation
2. **Event Streaming**: Real-time data processing
3. **Blockchain Integration**: Immutable event logs
4. **Machine Learning**: Event pattern analysis and prediction

This comprehensive project provides a solid foundation in modern distributed systems architecture and prepares you for building enterprise-scale applications!

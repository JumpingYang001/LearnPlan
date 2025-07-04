# Inter-Service Communication

*Duration: 2 weeks*

## Overview

Inter-service communication is the backbone of microservices architecture. Services must communicate with each other to fulfill business requirements, and choosing the right communication pattern is crucial for system performance, reliability, and maintainability.

This section covers the fundamental patterns, protocols, and best practices for enabling services to communicate effectively in a distributed system.

## Table of Contents

1. [Communication Patterns](#communication-patterns)
2. [Synchronous Communication](#synchronous-communication)
3. [Asynchronous Communication](#asynchronous-communication)
4. [REST API Design](#rest-api-design)
5. [gRPC Communication](#grpc-communication)
6. [Message-Based Communication](#message-based-communication)
7. [API Gateway Patterns](#api-gateway-patterns)
8. [Service Discovery](#service-discovery)
9. [Circuit Breaker Pattern](#circuit-breaker-pattern)
10. [Best Practices](#best-practices)

## Communication Patterns

### Overview of Communication Types

Inter-service communication can be categorized along two dimensions:

1. **Synchronous vs Asynchronous**
2. **One-to-One vs One-to-Many**

```
Communication Matrix:
                │ One-to-One        │ One-to-Many
─────────────────┼──────────────────┼─────────────────
Synchronous     │ Request/Response │ ---
Asynchronous    │ Notification     │ Publish/Subscribe
                │ Request/Async    │ Event Broadcasting
```

#### Visual Communication Flow

```
Synchronous Communication:
Service A ──────request──────> Service B
         <─────response──────

Asynchronous Communication:
Service A ──────message──────> Message Broker ──────> Service B
                                     │
                                     └─────> Service C
```

### Communication Pattern Decision Tree

```python
def choose_communication_pattern(requirements):
    """
    Decision helper for choosing communication patterns
    """
    if requirements.immediate_response_needed:
        if requirements.real_time_processing:
            return "Synchronous Request/Response"
        else:
            return "Asynchronous Request/Response"
    
    if requirements.multiple_consumers:
        if requirements.event_ordering_important:
            return "Event Streaming (Kafka)"
        else:
            return "Publish/Subscribe"
    
    if requirements.fire_and_forget:
        return "Message Queue"
    
    return "Request/Response with Caching"

# Example usage
requirements = {
    'immediate_response_needed': False,
    'multiple_consumers': True,
    'event_ordering_important': True,
    'fire_and_forget': False
}

pattern = choose_communication_pattern(requirements)
print(f"Recommended pattern: {pattern}")
```

## Synchronous Communication

### Characteristics

- **Blocking**: Caller waits for response
- **Direct Coupling**: Services are tightly coupled
- **Immediate Response**: Results available immediately
- **Error Handling**: Synchronous error propagation

### Request/Response Pattern

#### HTTP/REST Implementation

**Product Service (Flask)**
```python
# product_service.py
from flask import Flask, jsonify, request
from dataclasses import dataclass
from typing import List, Optional
import json

app = Flask(__name__)

@dataclass
class Product:
    id: int
    name: str
    price: float
    category: str
    stock: int

# In-memory database simulation
products_db = [
    Product(1, "Laptop", 999.99, "Electronics", 10),
    Product(2, "Book", 19.99, "Education", 50),
    Product(3, "Coffee", 4.99, "Food", 100)
]

@app.route('/products', methods=['GET'])
def get_products():
    """Get all products with optional filtering"""
    category = request.args.get('category')
    max_price = request.args.get('max_price', type=float)
    
    filtered_products = products_db
    
    if category:
        filtered_products = [p for p in filtered_products if p.category.lower() == category.lower()]
    
    if max_price:
        filtered_products = [p for p in filtered_products if p.price <= max_price]
    
    return jsonify([{
        'id': p.id,
        'name': p.name,
        'price': p.price,
        'category': p.category,
        'stock': p.stock
    } for p in filtered_products])

@app.route('/products/<int:product_id>', methods=['GET'])
def get_product(product_id):
    """Get specific product by ID"""
    product = next((p for p in products_db if p.id == product_id), None)
    
    if not product:
        return jsonify({'error': 'Product not found'}), 404
    
    return jsonify({
        'id': product.id,
        'name': product.name,
        'price': product.price,
        'category': product.category,
        'stock': product.stock
    })

@app.route('/products/<int:product_id>/stock', methods=['POST'])
def update_stock(product_id):
    """Update product stock"""
    data = request.get_json()
    quantity = data.get('quantity', 0)
    
    product = next((p for p in products_db if p.id == product_id), None)
    if not product:
        return jsonify({'error': 'Product not found'}), 404
    
    if product.stock + quantity < 0:
        return jsonify({'error': 'Insufficient stock'}), 400
    
    product.stock += quantity
    
    return jsonify({
        'product_id': product_id,
        'new_stock': product.stock,
        'change': quantity
    })

if __name__ == '__main__':
    app.run(port=5001, debug=True)
```

**Order Service Client**
```python
# order_service.py
import requests
import json
from dataclasses import dataclass
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OrderItem:
    product_id: int
    quantity: int
    price: float

@dataclass
class Order:
    order_id: str
    customer_id: str
    items: List[OrderItem]
    total: float
    status: str

class ProductServiceClient:
    """Client for communicating with Product Service"""
    
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'OrderService/1.0'
        })
    
    def get_product(self, product_id: int) -> Optional[dict]:
        """Get product details"""
        try:
            response = self.session.get(
                f"{self.base_url}/products/{product_id}",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get product {product_id}: {e}")
            return None
    
    def check_stock(self, product_id: int, required_quantity: int) -> bool:
        """Check if product has sufficient stock"""
        product = self.get_product(product_id)
        if not product:
            return False
        
        return product['stock'] >= required_quantity
    
    def reserve_stock(self, product_id: int, quantity: int) -> bool:
        """Reserve stock for order"""
        try:
            response = self.session.post(
                f"{self.base_url}/products/{product_id}/stock",
                json={'quantity': -quantity},
                timeout=5
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to reserve stock for product {product_id}: {e}")
            return False

class OrderService:
    """Order service that communicates with Product service"""
    
    def __init__(self):
        self.product_client = ProductServiceClient()
        self.orders_db = {}
    
    def create_order(self, customer_id: str, order_items: List[dict]) -> dict:
        """Create a new order with stock validation"""
        order_id = f"ORDER-{len(self.orders_db) + 1:04d}"
        
        # Step 1: Validate all products and stock
        validated_items = []
        total_amount = 0
        
        for item in order_items:
            product_id = item['product_id']
            quantity = item['quantity']
            
            # Get product details
            product = self.product_client.get_product(product_id)
            if not product:
                return {'error': f'Product {product_id} not found'}
            
            # Check stock availability
            if not self.product_client.check_stock(product_id, quantity):
                return {'error': f'Insufficient stock for product {product_id}'}
            
            validated_items.append(OrderItem(
                product_id=product_id,
                quantity=quantity,
                price=product['price']
            ))
            total_amount += product['price'] * quantity
        
        # Step 2: Reserve stock for all items
        reserved_products = []
        try:
            for item in validated_items:
                if self.product_client.reserve_stock(item.product_id, item.quantity):
                    reserved_products.append(item.product_id)
                else:
                    # Rollback previous reservations
                    self._rollback_reservations(reserved_products, validated_items)
                    return {'error': f'Failed to reserve stock for product {item.product_id}'}
            
            # Step 3: Create order
            order = Order(
                order_id=order_id,
                customer_id=customer_id,
                items=validated_items,
                total=total_amount,
                status='confirmed'
            )
            
            self.orders_db[order_id] = order
            
            logger.info(f"Order {order_id} created successfully")
            return {
                'order_id': order_id,
                'total': total_amount,
                'status': 'confirmed',
                'items': len(validated_items)
            }
            
        except Exception as e:
            # Rollback all reservations
            self._rollback_reservations(reserved_products, validated_items)
            logger.error(f"Order creation failed: {e}")
            return {'error': 'Order creation failed'}
    
    def _rollback_reservations(self, reserved_products: List[int], items: List[OrderItem]):
        """Rollback stock reservations"""
        for product_id in reserved_products:
            item = next(item for item in items if item.product_id == product_id)
            # Return stock (positive quantity)
            self.product_client.reserve_stock(product_id, -item.quantity)

# Example usage
if __name__ == '__main__':
    order_service = OrderService()
    
    # Create an order
    order_items = [
        {'product_id': 1, 'quantity': 2},
        {'product_id': 3, 'quantity': 5}
    ]
    
    result = order_service.create_order('CUSTOMER-001', order_items)
    print(f"Order result: {result}")
```

### Advantages and Disadvantages

**Advantages:**
- Simple to understand and implement
- Immediate feedback and error handling
- Strong consistency guarantees
- Easy debugging and tracing

**Disadvantages:**
- Creates tight coupling between services
- Cascade failures (if one service fails, others fail)
- Increased latency due to blocking calls
- Reduced system availability

### Synchronous Communication Best Practices

```python
# best_practices.py
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import logging
from contextlib import contextmanager

class ResilientHTTPClient:
    """HTTP client with retry, timeout, and circuit breaker patterns"""
    
    def __init__(self, base_url: str, timeout: int = 5, max_retries: int = 3):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    @contextmanager
    def timeout_context(self, custom_timeout: int = None):
        """Context manager for custom timeout"""
        original_timeout = self.timeout
        if custom_timeout:
            self.timeout = custom_timeout
        try:
            yield
        finally:
            self.timeout = original_timeout
    
    def get(self, endpoint: str, **kwargs):
        """GET request with built-in resilience"""
        return self._make_request('GET', endpoint, **kwargs)
    
    def post(self, endpoint: str, **kwargs):
        """POST request with built-in resilience"""
        return self._make_request('POST', endpoint, **kwargs)
    
    def _make_request(self, method: str, endpoint: str, **kwargs):
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response
            
        except requests.exceptions.Timeout:
            logging.error(f"Timeout for {method} {url}")
            raise
        except requests.exceptions.ConnectionError:
            logging.error(f"Connection error for {method} {url}")
            raise
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error for {method} {url}: {e}")
            raise

# Usage example
client = ResilientHTTPClient("http://localhost:5001")

# Use with custom timeout
with client.timeout_context(10):
    response = client.get("/products/1")
    print(response.json())
```

## Asynchronous Communication

### Characteristics

- **Non-blocking**: Caller doesn't wait for response
- **Loose Coupling**: Services are decoupled
- **Eventually Consistent**: Data consistency achieved over time
- **Higher Throughput**: Better resource utilization

### Message Queue Pattern

#### Using Redis as Message Broker

**Publisher Service**
```python
# publisher_service.py
import redis
import json
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Event:
    event_id: str
    event_type: str
    source_service: str
    timestamp: str
    data: Dict[Any, Any]
    correlation_id: str = None

class EventPublisher:
    """Publishes events to Redis message broker"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.service_name = "order-service"
    
    def publish_event(self, event_type: str, data: Dict[Any, Any], correlation_id: str = None):
        """Publish an event to the message broker"""
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source_service=self.service_name,
            timestamp=datetime.utcnow().isoformat(),
            data=data,
            correlation_id=correlation_id or str(uuid.uuid4())
        )
        
        try:
            # Publish to specific channel
            channel = f"events.{event_type}"
            message = json.dumps(asdict(event))
            
            result = self.redis_client.publish(channel, message)
            logger.info(f"Published event {event.event_id} to channel {channel}. Subscribers: {result}")
            
            # Also store in a list for durability (optional)
            self.redis_client.lpush(f"events:{event_type}:history", message)
            
            return event.event_id
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            raise

# Order service that publishes events
class AsyncOrderService:
    """Order service that publishes events asynchronously"""
    
    def __init__(self):
        self.event_publisher = EventPublisher()
        self.orders_db = {}
    
    def create_order(self, customer_id: str, order_items: List[dict]) -> dict:
        """Create order and publish events"""
        order_id = f"ORDER-{len(self.orders_db) + 1:04d}"
        
        # Create order (simplified - no sync validation)
        order = {
            'order_id': order_id,
            'customer_id': customer_id,
            'items': order_items,
            'status': 'pending',
            'created_at': datetime.utcnow().isoformat()
        }
        
        self.orders_db[order_id] = order
        
        # Publish order created event
        self.event_publisher.publish_event(
            event_type='order.created',
            data=order
        )
        
        logger.info(f"Order {order_id} created and event published")
        return {'order_id': order_id, 'status': 'pending'}
    
    def confirm_order(self, order_id: str):
        """Confirm order after stock validation"""
        if order_id in self.orders_db:
            self.orders_db[order_id]['status'] = 'confirmed'
            
            # Publish order confirmed event
            self.event_publisher.publish_event(
                event_type='order.confirmed',
                data={'order_id': order_id, 'status': 'confirmed'}
            )
    
    def cancel_order(self, order_id: str, reason: str):
        """Cancel order"""
        if order_id in self.orders_db:
            self.orders_db[order_id]['status'] = 'cancelled'
            self.orders_db[order_id]['cancellation_reason'] = reason
            
            # Publish order cancelled event
            self.event_publisher.publish_event(
                event_type='order.cancelled',
                data={
                    'order_id': order_id, 
                    'status': 'cancelled',
                    'reason': reason
                }
            )

# Example usage
if __name__ == '__main__':
    order_service = AsyncOrderService()
    
    # Create order
    result = order_service.create_order('CUSTOMER-001', [
        {'product_id': 1, 'quantity': 2}
    ])
    print(f"Order created: {result}")
```

**Subscriber Service**
```python
# subscriber_service.py
import redis
import json
import logging
import threading
import time
from typing import Callable, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventSubscriber:
    """Subscribes to events from Redis message broker"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.pubsub = self.redis_client.pubsub()
        self.handlers: Dict[str, Callable] = {}
        self.running = False
    
    def subscribe_to_event(self, event_type: str, handler: Callable):
        """Subscribe to specific event type"""
        channel = f"events.{event_type}"
        self.pubsub.subscribe(channel)
        self.handlers[channel] = handler
        logger.info(f"Subscribed to {channel}")
    
    def start_listening(self):
        """Start listening for events"""
        self.running = True
        logger.info("Started listening for events...")
        
        try:
            for message in self.pubsub.listen():
                if not self.running:
                    break
                
                if message['type'] == 'message':
                    self._handle_message(message)
                    
        except KeyboardInterrupt:
            logger.info("Stopping event listener...")
        finally:
            self.stop_listening()
    
    def stop_listening(self):
        """Stop listening for events"""
        self.running = False
        self.pubsub.close()
    
    def _handle_message(self, message):
        """Handle incoming message"""
        try:
            channel = message['channel']
            data = json.loads(message['data'])
            
            if channel in self.handlers:
                handler = self.handlers[channel]
                handler(data)
            else:
                logger.warning(f"No handler for channel {channel}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")

# Inventory service that subscribes to order events
class InventoryService:
    """Inventory service that handles order events"""
    
    def __init__(self):
        self.event_subscriber = EventSubscriber()
        self.inventory_db = {
            1: {'product_id': 1, 'stock': 100},
            2: {'product_id': 2, 'stock': 50},
            3: {'product_id': 3, 'stock': 200}
        }
        
        # Subscribe to order events
        self.event_subscriber.subscribe_to_event('order.created', self.handle_order_created)
        self.event_subscriber.subscribe_to_event('order.cancelled', self.handle_order_cancelled)
    
    def handle_order_created(self, event_data: dict):
        """Handle order created event"""
        logger.info(f"Processing order created event: {event_data['event_id']}")
        
        order_data = event_data['data']
        order_id = order_data['order_id']
        
        # Validate stock for all items
        can_fulfill = True
        stock_requirements = {}
        
        for item in order_data['items']:
            product_id = item['product_id']
            quantity = item['quantity']
            
            if product_id not in self.inventory_db:
                logger.error(f"Product {product_id} not found")
                can_fulfill = False
                break
            
            available_stock = self.inventory_db[product_id]['stock']
            if available_stock < quantity:
                logger.error(f"Insufficient stock for product {product_id}")
                can_fulfill = False
                break
            
            stock_requirements[product_id] = quantity
        
        if can_fulfill:
            # Reserve stock
            for product_id, quantity in stock_requirements.items():
                self.inventory_db[product_id]['stock'] -= quantity
                logger.info(f"Reserved {quantity} units of product {product_id}")
            
            # Publish stock reserved event
            self._publish_stock_reserved_event(order_id, stock_requirements)
        else:
            # Publish stock unavailable event
            self._publish_stock_unavailable_event(order_id)
    
    def handle_order_cancelled(self, event_data: dict):
        """Handle order cancelled event"""
        logger.info(f"Processing order cancelled event: {event_data['event_id']}")
        
        # In a real system, you'd restore the reserved stock
        # This is a simplified implementation
        order_id = event_data['data']['order_id']
        logger.info(f"Order {order_id} cancelled - stock would be restored")
    
    def _publish_stock_reserved_event(self, order_id: str, stock_requirements: dict):
        """Publish stock reserved event"""
        # In a real system, this service would also be a publisher
        logger.info(f"Stock reserved for order {order_id}: {stock_requirements}")
    
    def _publish_stock_unavailable_event(self, order_id: str):
        """Publish stock unavailable event"""
        logger.info(f"Stock unavailable for order {order_id}")
    
    def start(self):
        """Start the inventory service"""
        logger.info("Starting Inventory Service...")
        self.event_subscriber.start_listening()

# Example usage
if __name__ == '__main__':
    inventory_service = InventoryService()
    
    # Start in a separate thread to allow for graceful shutdown
    listener_thread = threading.Thread(target=inventory_service.start)
    listener_thread.daemon = True
    listener_thread.start()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        inventory_service.event_subscriber.stop_listening()
        logger.info("Inventory service stopped")
```

### Event Sourcing Pattern

```python
# event_sourcing.py
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from datetime import datetime
import json
import uuid

@dataclass
class Event:
    aggregate_id: str
    event_type: str
    event_data: Dict[str, Any]
    event_version: int
    timestamp: datetime
    event_id: str = None
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())

class EventStore:
    """Simple in-memory event store"""
    
    def __init__(self):
        self.events: List[Event] = []
        self.snapshots: Dict[str, Dict] = {}
    
    def append_event(self, event: Event):
        """Append event to the store"""
        self.events.append(event)
    
    def get_events(self, aggregate_id: str, from_version: int = 0) -> List[Event]:
        """Get events for an aggregate"""
        return [
            event for event in self.events 
            if event.aggregate_id == aggregate_id and event.event_version > from_version
        ]
    
    def save_snapshot(self, aggregate_id: str, state: Dict, version: int):
        """Save aggregate snapshot"""
        self.snapshots[aggregate_id] = {
            'state': state,
            'version': version,
            'timestamp': datetime.utcnow()
        }

class OrderAggregate:
    """Order aggregate using event sourcing"""
    
    def __init__(self, order_id: str):
        self.order_id = order_id
        self.version = 0
        self.status = 'new'
        self.items = []
        self.customer_id = None
        self.total_amount = 0
        self.events: List[Event] = []
    
    def create_order(self, customer_id: str, items: List[dict]):
        """Create order command"""
        if self.status != 'new':
            raise ValueError("Order already exists")
        
        # Calculate total
        total = sum(item['price'] * item['quantity'] for item in items)
        
        event = Event(
            aggregate_id=self.order_id,
            event_type='OrderCreated',
            event_data={
                'customer_id': customer_id,
                'items': items,
                'total_amount': total
            },
            event_version=self.version + 1,
            timestamp=datetime.utcnow()
        )
        
        self._apply_event(event)
        self.events.append(event)
    
    def confirm_order(self):
        """Confirm order command"""
        if self.status != 'pending':
            raise ValueError("Order cannot be confirmed")
        
        event = Event(
            aggregate_id=self.order_id,
            event_type='OrderConfirmed',
            event_data={'confirmed_at': datetime.utcnow().isoformat()},
            event_version=self.version + 1,
            timestamp=datetime.utcnow()
        )
        
        self._apply_event(event)
        self.events.append(event)
    
    def cancel_order(self, reason: str):
        """Cancel order command"""
        if self.status in ['shipped', 'delivered']:
            raise ValueError("Order cannot be cancelled")
        
        event = Event(
            aggregate_id=self.order_id,
            event_type='OrderCancelled',
            event_data={
                'reason': reason,
                'cancelled_at': datetime.utcnow().isoformat()
            },
            event_version=self.version + 1,
            timestamp=datetime.utcnow()
        )
        
        self._apply_event(event)
        self.events.append(event)
    
    def _apply_event(self, event: Event):
        """Apply event to aggregate state"""
        if event.event_type == 'OrderCreated':
            self.customer_id = event.event_data['customer_id']
            self.items = event.event_data['items']
            self.total_amount = event.event_data['total_amount']
            self.status = 'pending'
        
        elif event.event_type == 'OrderConfirmed':
            self.status = 'confirmed'
        
        elif event.event_type == 'OrderCancelled':
            self.status = 'cancelled'
        
        self.version = event.event_version
    
    def load_from_events(self, events: List[Event]):
        """Rebuild aggregate state from events"""
        for event in sorted(events, key=lambda e: e.event_version):
            self._apply_event(event)
    
    def get_uncommitted_events(self) -> List[Event]:
        """Get events that haven't been persisted"""
        return self.events.copy()
    
    def mark_events_as_committed(self):
        """Mark events as persisted"""
        self.events.clear()

# Usage example
if __name__ == '__main__':
    event_store = EventStore()
    
    # Create order aggregate
    order = OrderAggregate('ORDER-001')
    
    # Execute commands
    order.create_order('CUSTOMER-001', [
        {'product_id': 1, 'quantity': 2, 'price': 999.99}
    ])
    
    order.confirm_order()
    
    # Persist events
    for event in order.get_uncommitted_events():
        event_store.append_event(event)
    
    order.mark_events_as_committed()
    
    # Rebuild aggregate from events
    new_order = OrderAggregate('ORDER-001')
    events = event_store.get_events('ORDER-001')
    new_order.load_from_events(events)
    
    print(f"Rebuilt order status: {new_order.status}")
    print(f"Order total: {new_order.total_amount}")
```

## REST API Design

### RESTful Principles

REST (Representational State Transfer) is an architectural style for designing web services based on these principles:

1. **Stateless**: Each request contains all information needed
2. **Client-Server**: Clear separation of concerns
3. **Cacheable**: Responses should indicate cacheability
4. **Uniform Interface**: Consistent resource-based URLs
5. **Layered System**: Architecture can be composed of hierarchical layers

### Resource-Based URL Design

```python
# Good RESTful URL patterns
GET    /api/v1/products              # Get all products
GET    /api/v1/products/123          # Get specific product
POST   /api/v1/products              # Create new product
PUT    /api/v1/products/123          # Update entire product
PATCH  /api/v1/products/123          # Partial update
DELETE /api/v1/products/123          # Delete product

# Nested resources
GET    /api/v1/products/123/reviews  # Get reviews for product
POST   /api/v1/products/123/reviews  # Add review to product
GET    /api/v1/reviews/456           # Get specific review

# Query parameters for filtering, sorting, pagination
GET    /api/v1/products?category=electronics&sort=price&page=2&limit=20
```

### Comprehensive REST API Implementation

```python
# advanced_rest_api.py
from flask import Flask, request, jsonify, abort
from flask_sqlalchemy import SQLAlchemy
from marshmallow import Schema, fields, ValidationError
from werkzeug.exceptions import HTTPException
import logging
from datetime import datetime
from typing import Dict, List, Optional
import uuid

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///products.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    price = db.Column(db.Float, nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('category.id'))
    stock = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    category = db.relationship('Category', backref='products')
    reviews = db.relationship('Review', backref='product', cascade='all, delete-orphan')

class Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    description = db.Column(db.Text)

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    customer_id = db.Column(db.String(50), nullable=False)
    rating = db.Column(db.Integer, nullable=False)  # 1-5 stars
    comment = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Schemas for serialization/validation
class CategorySchema(Schema):
    id = fields.Int(dump_only=True)
    name = fields.Str(required=True, validate=lambda x: len(x) <= 50)
    description = fields.Str()

class ProductSchema(Schema):
    id = fields.Int(dump_only=True)
    name = fields.Str(required=True, validate=lambda x: len(x) <= 100)
    description = fields.Str()
    price = fields.Float(required=True, validate=lambda x: x > 0)
    category_id = fields.Int(required=True)
    stock = fields.Int(validate=lambda x: x >= 0)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)
    
    # Nested fields
    category = fields.Nested(CategorySchema, dump_only=True)

class ReviewSchema(Schema):
    id = fields.Int(dump_only=True)
    product_id = fields.Int(required=True)
    customer_id = fields.Str(required=True)
    rating = fields.Int(required=True, validate=lambda x: 1 <= x <= 5)
    comment = fields.Str()
    created_at = fields.DateTime(dump_only=True)

# Initialize schemas
product_schema = ProductSchema()
products_schema = ProductSchema(many=True)
category_schema = CategorySchema()
categories_schema = CategorySchema(many=True)
review_schema = ReviewSchema()
reviews_schema = ReviewSchema(many=True)

# Error handlers
@app.errorhandler(ValidationError)
def handle_validation_error(e):
    return jsonify({'error': 'Validation failed', 'messages': e.messages}), 400

@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def handle_internal_error(e):
    logger.error(f"Internal error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

# Utility functions
def paginate_query(query, page: int = 1, per_page: int = 20):
    """Paginate a SQLAlchemy query"""
    return query.paginate(
        page=page, 
        per_page=per_page, 
        error_out=False
    )

def build_response(data, status_code: int = 200, pagination=None):
    """Build standardized response"""
    response = {'data': data}
    
    if pagination:
        response['pagination'] = {
            'page': pagination.page,
            'pages': pagination.pages,
            'per_page': pagination.per_page,
            'total': pagination.total,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev
        }
    
    return jsonify(response), status_code

# Routes
@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })

# Categories endpoints
@app.route('/api/v1/categories', methods=['GET'])
def get_categories():
    """Get all categories"""
    categories = Category.query.all()
    return build_response(categories_schema.dump(categories))

@app.route('/api/v1/categories/<int:category_id>', methods=['GET'])
def get_category(category_id):
    """Get specific category"""
    category = Category.query.get_or_404(category_id)
    return build_response(category_schema.dump(category))

@app.route('/api/v1/categories', methods=['POST'])
def create_category():
    """Create new category"""
    try:
        data = category_schema.load(request.json)
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'messages': e.messages}), 400
    
    category = Category(**data)
    db.session.add(category)
    
    try:
        db.session.commit()
        logger.info(f"Created category: {category.name}")
        return build_response(category_schema.dump(category), 201)
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to create category: {e}")
        return jsonify({'error': 'Failed to create category'}), 500

# Products endpoints
@app.route('/api/v1/products', methods=['GET'])
def get_products():
    """Get products with filtering, sorting, and pagination"""
    # Query parameters
    category_id = request.args.get('category_id', type=int)
    min_price = request.args.get('min_price', type=float)
    max_price = request.args.get('max_price', type=float)
    search = request.args.get('search')
    sort_by = request.args.get('sort', 'created_at')
    sort_order = request.args.get('order', 'desc')
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)  # Max 100 items per page
    
    # Build query
    query = Product.query.join(Category)
    
    # Apply filters
    if category_id:
        query = query.filter(Product.category_id == category_id)
    
    if min_price is not None:
        query = query.filter(Product.price >= min_price)
    
    if max_price is not None:
        query = query.filter(Product.price <= max_price)
    
    if search:
        search_pattern = f"%{search}%"
        query = query.filter(
            db.or_(
                Product.name.ilike(search_pattern),
                Product.description.ilike(search_pattern)
            )
        )
    
    # Apply sorting
    sort_column = getattr(Product, sort_by, Product.created_at)
    if sort_order.lower() == 'desc':
        query = query.order_by(sort_column.desc())
    else:
        query = query.order_by(sort_column.asc())
    
    # Paginate
    pagination = paginate_query(query, page, per_page)
    products = pagination.items
    
    return build_response(products_schema.dump(products), pagination=pagination)

@app.route('/api/v1/products/<int:product_id>', methods=['GET'])
def get_product(product_id):
    """Get specific product with reviews"""
    product = Product.query.get_or_404(product_id)
    product_data = product_schema.dump(product)
    
    # Include reviews in response
    reviews = Review.query.filter_by(product_id=product_id).order_by(Review.created_at.desc()).all()
    product_data['reviews'] = reviews_schema.dump(reviews)
    
    # Calculate average rating
    if reviews:
        avg_rating = sum(review.rating for review in reviews) / len(reviews)
        product_data['average_rating'] = round(avg_rating, 2)
        product_data['review_count'] = len(reviews)
    
    return build_response(product_data)

@app.route('/api/v1/products', methods=['POST'])
def create_product():
    """Create new product"""
    try:
        data = product_schema.load(request.json)
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'messages': e.messages}), 400
    
    # Verify category exists
    category = Category.query.get(data['category_id'])
    if not category:
        return jsonify({'error': 'Category not found'}), 400
    
    product = Product(**data)
    db.session.add(product)
    
    try:
        db.session.commit()
        logger.info(f"Created product: {product.name}")
        return build_response(product_schema.dump(product), 201)
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to create product: {e}")
        return jsonify({'error': 'Failed to create product'}), 500

@app.route('/api/v1/products/<int:product_id>', methods=['PUT'])
def update_product(product_id):
    """Update entire product"""
    product = Product.query.get_or_404(product_id)
    
    try:
        data = product_schema.load(request.json)
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'messages': e.messages}), 400
    
    # Update product fields
    for key, value in data.items():
        setattr(product, key, value)
    
    product.updated_at = datetime.utcnow()
    
    try:
        db.session.commit()
        logger.info(f"Updated product: {product.name}")
        return build_response(product_schema.dump(product))
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to update product: {e}")
        return jsonify({'error': 'Failed to update product'}), 500

@app.route('/api/v1/products/<int:product_id>', methods=['PATCH'])
def partial_update_product(product_id):
    """Partially update product"""
    product = Product.query.get_or_404(product_id)
    
    try:
        # Allow partial updates by using partial=True
        data = product_schema.load(request.json, partial=True)
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'messages': e.messages}), 400
    
    # Update only provided fields
    for key, value in data.items():
        setattr(product, key, value)
    
    product.updated_at = datetime.utcnow()
    
    try:
        db.session.commit()
        logger.info(f"Partially updated product: {product.name}")
        return build_response(product_schema.dump(product))
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to update product: {e}")
        return jsonify({'error': 'Failed to update product'}), 500

@app.route('/api/v1/products/<int:product_id>', methods=['DELETE'])
def delete_product(product_id):
    """Delete product"""
    product = Product.query.get_or_404(product_id)
    
    try:
        db.session.delete(product)
        db.session.commit()
        logger.info(f"Deleted product: {product.name}")
        return '', 204
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to delete product: {e}")
        return jsonify({'error': 'Failed to delete product'}), 500

# Reviews endpoints (nested resource)
@app.route('/api/v1/products/<int:product_id>/reviews', methods=['GET'])
def get_product_reviews(product_id):
    """Get reviews for a specific product"""
    # Verify product exists
    Product.query.get_or_404(product_id)
    
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 10, type=int), 50)
    
    query = Review.query.filter_by(product_id=product_id).order_by(Review.created_at.desc())
    pagination = paginate_query(query, page, per_page)
    
    return build_response(reviews_schema.dump(pagination.items), pagination=pagination)

@app.route('/api/v1/products/<int:product_id>/reviews', methods=['POST'])
def create_review(product_id):
    """Create review for a product"""
    # Verify product exists
    Product.query.get_or_404(product_id)
    
    try:
        data = review_schema.load(request.json)
        data['product_id'] = product_id  # Override with URL parameter
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'messages': e.messages}), 400
    
    # Check if customer already reviewed this product
    existing_review = Review.query.filter_by(
        product_id=product_id,
        customer_id=data['customer_id']
    ).first()
    
    if existing_review:
        return jsonify({'error': 'Customer has already reviewed this product'}), 409
    
    review = Review(**data)
    db.session.add(review)
    
    try:
        db.session.commit()
        logger.info(f"Created review for product {product_id}")
        return build_response(review_schema.dump(review), 201)
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to create review: {e}")
        return jsonify({'error': 'Failed to create review'}), 500

# API versioning and content negotiation
@app.before_request
def before_request():
    """Request preprocessing"""
    # Log request
    logger.info(f"{request.method} {request.path} - {request.remote_addr}")
    
    # Validate Content-Type for POST/PUT/PATCH
    if request.method in ['POST', 'PUT', 'PATCH']:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400

@app.after_request
def after_request(response):
    """Response postprocessing"""
    # Add CORS headers
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
    
    # Add API version header
    response.headers.add('API-Version', '1.0')
    
    return response

# Initialize database
@app.before_first_request
def create_tables():
    db.create_all()
    
    # Create sample data
    if not Category.query.first():
        categories = [
            Category(name='Electronics', description='Electronic devices and gadgets'),
            Category(name='Books', description='Books and educational materials'),
            Category(name='Clothing', description='Apparel and accessories')
        ]
        
        for category in categories:
            db.session.add(category)
        
        db.session.commit()

if __name__ == '__main__':
    app.run(debug=True, port=5001)
```

### REST API Client Implementation

```python
# rest_client.py
import requests
from typing import Dict, List, Optional, Any
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductAPIClient:
    """Client for interacting with Product REST API"""
    
    def __init__(self, base_url: str = "http://localhost:5001/api/v1", timeout: int = 10):
        self.base_url = base_url
        self.timeout = timeout
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Default headers
        session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'ProductAPIClient/1.0'
        })
        
        return session
    
    def get_products(self, **filters) -> Dict[str, Any]:
        """Get products with optional filters"""
        try:
            response = self.session.get(
                f"{self.base_url}/products",
                params=filters,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching products: {e}")
            raise
    
    def get_product(self, product_id: int) -> Dict[str, Any]:
        """Get specific product"""
        try:
            response = self.session.get(
                f"{self.base_url}/products/{product_id}",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching product {product_id}: {e}")
            raise
    
    def create_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new product"""
        try:
            response = self.session.post(
                f"{self.base_url}/products",
                json=product_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating product: {e}")
            raise
    
    def update_product(self, product_id: int, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update entire product"""
        try:
            response = self.session.put(
                f"{self.base_url}/products/{product_id}",
                json=product_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error updating product {product_id}: {e}")
            raise
    
    def partial_update_product(self, product_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Partially update product"""
        try:
            response = self.session.patch(
                f"{self.base_url}/products/{product_id}",
                json=updates,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error partially updating product {product_id}: {e}")
            raise
    
    def delete_product(self, product_id: int) -> bool:
        """Delete product"""
        try:
            response = self.session.delete(
                f"{self.base_url}/products/{product_id}",
                timeout=self.timeout
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error deleting product {product_id}: {e}")
            raise
    
    def create_review(self, product_id: int, review_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create review for product"""
        try:
            response = self.session.post(
                f"{self.base_url}/products/{product_id}/reviews",
                json=review_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating review for product {product_id}: {e}")
            raise

# Example usage
if __name__ == '__main__':
    client = ProductAPIClient()
    
    try:
        # Get all products
        products = client.get_products(category_id=1, max_price=1000)
        print(f"Found {len(products['data'])} products")
        
        # Get specific product
        if products['data']:
            product_id = products['data'][0]['id']
            product = client.get_product(product_id)
            print(f"Product: {product['data']['name']}")
        
        # Create new product
        new_product = {
            'name': 'Test Product',
            'description': 'A test product',
            'price': 99.99,
            'category_id': 1,
            'stock': 10
        }
        
        created = client.create_product(new_product)
        print(f"Created product with ID: {created['data']['id']}")
        
    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")
```

### RESTful Best Practices

1. **Use HTTP Status Codes Correctly**
```python
# Status code guidelines
200 # OK - Successful GET, PUT, PATCH
201 # Created - Successful POST
204 # No Content - Successful DELETE
400 # Bad Request - Invalid request data
401 # Unauthorized - Authentication required
403 # Forbidden - Access denied
404 # Not Found - Resource doesn't exist
409 # Conflict - Resource conflict (duplicate)
422 # Unprocessable Entity - Validation errors
500 # Internal Server Error - Server errors
```

2. **Implement Proper Error Handling**
```python
# Standardized error response format
{
    "error": {
        "code": "VALIDATION_FAILED",
        "message": "Request validation failed",
        "details": {
            "field": "price",
            "reason": "must be greater than 0"
        },
        "timestamp": "2024-01-01T12:00:00Z",
        "request_id": "uuid-123"
    }
}
```

3. **Use HATEOAS (Hypermedia as Engine of Application State)**
```python
# Example response with links
{
    "data": {
        "id": 123,
        "name": "Product Name",
        "price": 99.99
    },
    "links": {
        "self": "/api/v1/products/123",
        "reviews": "/api/v1/products/123/reviews",
        "category": "/api/v1/categories/1",
        "edit": "/api/v1/products/123",
        "delete": "/api/v1/products/123"
    }
}
```

## gRPC Communication

### Introduction to gRPC

gRPC is a high-performance, open-source universal RPC framework that uses HTTP/2 and Protocol Buffers by default. It's particularly well-suited for:

- High-performance inter-service communication
- Polyglot environments (multiple programming languages)
- Streaming data
- Type-safe APIs

### Protocol Buffers Definition

```protobuf
// product_service.proto
syntax = "proto3";

package product;

// Product message
message Product {
    int32 id = 1;
    string name = 2;
    string description = 3;
    double price = 4;
    int32 category_id = 5;
    int32 stock = 6;
    string created_at = 7;
    string updated_at = 8;
}

// Request messages
message GetProductRequest {
    int32 id = 1;
}

message GetProductsRequest {
    int32 category_id = 1;
    double min_price = 2;
    double max_price = 3;
    string search = 4;
    int32 page = 5;
    int32 page_size = 6;
}

message CreateProductRequest {
    string name = 1;
    string description = 2;
    double price = 3;
    int32 category_id = 4;
    int32 stock = 5;
}

message UpdateProductRequest {
    int32 id = 1;
    string name = 2;
    string description = 3;
    double price = 4;
    int32 category_id = 5;
    int32 stock = 6;
}

message DeleteProductRequest {
    int32 id = 1;
}

// Response messages
message GetProductResponse {
    Product product = 1;
    bool found = 2;
}

message GetProductsResponse {
    repeated Product products = 1;
    int32 total_count = 2;
    int32 page = 3;
    int32 page_size = 4;
}

message CreateProductResponse {
    Product product = 1;
    bool success = 2;
    string error_message = 3;
}

message UpdateProductResponse {
    Product product = 1;
    bool success = 2;
    string error_message = 3;
}

message DeleteProductResponse {
    bool success = 1;
    string error_message = 2;
}

// Streaming messages
message ProductUpdateEvent {
    string event_type = 1; // created, updated, deleted
    Product product = 2;
    string timestamp = 3;
}

message SubscribeToUpdatesRequest {
    repeated int32 category_ids = 1;
}

// Service definition
service ProductService {
    // Unary RPC
    rpc GetProduct(GetProductRequest) returns (GetProductResponse);
    rpc GetProducts(GetProductsRequest) returns (GetProductsResponse);
    rpc CreateProduct(CreateProductRequest) returns (CreateProductResponse);
    rpc UpdateProduct(UpdateProductRequest) returns (UpdateProductResponse);
    rpc DeleteProduct(DeleteProductRequest) returns (DeleteProductResponse);
    
    // Server streaming
    rpc SubscribeToProductUpdates(SubscribeToUpdatesRequest) returns (stream ProductUpdateEvent);
    
    // Client streaming
    rpc BatchCreateProducts(stream CreateProductRequest) returns (CreateProductResponse);
    
    // Bidirectional streaming
    rpc ProductChat(stream ProductUpdateEvent) returns (stream ProductUpdateEvent);
}
```

### gRPC Server Implementation

```python
# grpc_server.py
import grpc
from concurrent import futures
import threading
import time
import logging
from typing import Dict, List
import json
from datetime import datetime

# Generated from protobuf
import product_service_pb2
import product_service_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductServiceImpl(product_service_pb2_grpc.ProductServiceServicer):
    """gRPC Product Service implementation"""
    
    def __init__(self):
        # In-memory database simulation
        self.products_db: Dict[int, dict] = {
            1: {
                'id': 1,
                'name': 'Laptop',
                'description': 'High-performance laptop',
                'price': 999.99,
                'category_id': 1,
                'stock': 10,
                'created_at': '2024-01-01T00:00:00Z',
                'updated_at': '2024-01-01T00:00:00Z'
            },
            2: {
                'id': 2,
                'name': 'Mouse',
                'description': 'Wireless mouse',
                'price': 29.99,
                'category_id': 1,
                'stock': 50,
                'created_at': '2024-01-01T00:00:00Z',
                'updated_at': '2024-01-01T00:00:00Z'
            }
        }
        self.next_id = 3
        self.subscribers = []  # For streaming
        self.lock = threading.Lock()
    
    def GetProduct(self, request, context):
        """Get single product"""
        logger.info(f"GetProduct called for ID: {request.id}")
        
        with self.lock:
            if request.id in self.products_db:
                product_data = self.products_db[request.id]
                product = self._dict_to_product(product_data)
                return product_service_pb2.GetProductResponse(
                    product=product,
                    found=True
                )
            else:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f'Product with ID {request.id} not found')
                return product_service_pb2.GetProductResponse(found=False)
    
    def GetProducts(self, request, context):
        """Get products with filtering"""
        logger.info(f"GetProducts called with filters: category_id={request.category_id}")
        
        with self.lock:
            products = list(self.products_db.values())
        
        # Apply filters
        if request.category_id > 0:
            products = [p for p in products if p['category_id'] == request.category_id]
        
        if request.min_price > 0:
            products = [p for p in products if p['price'] >= request.min_price]
        
        if request.max_price > 0:
            products = [p for p in products if p['price'] <= request.max_price]
        
        if request.search:
            search_lower = request.search.lower()
            products = [
                p for p in products 
                if search_lower in p['name'].lower() or search_lower in p['description'].lower()
            ]
        
        # Pagination
        page = max(1, request.page)
        page_size = min(max(1, request.page_size), 100)  # Max 100 items
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        paginated_products = products[start_idx:end_idx]
        
        # Convert to protobuf
        pb_products = [self._dict_to_product(p) for p in paginated_products]
        
        return product_service_pb2.GetProductsResponse(
            products=pb_products,
            total_count=len(products),
            page=page,
            page_size=page_size
        )
    
    def CreateProduct(self, request, context):
        """Create new product"""
        logger.info(f"CreateProduct called: {request.name}")
        
        # Validation
        if not request.name or request.price <= 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('Invalid product data')
            return product_service_pb2.CreateProductResponse(
                success=False,
                error_message='Invalid product data'
            )
        
        with self.lock:
            product_data = {
                'id': self.next_id,
                'name': request.name,
                'description': request.description,
                'price': request.price,
                'category_id': request.category_id,
                'stock': request.stock,
                'created_at': datetime.utcnow().isoformat() + 'Z',
                'updated_at': datetime.utcnow().isoformat() + 'Z'
            }
            
            self.products_db[self.next_id] = product_data
            product_id = self.next_id
            self.next_id += 1
        
        # Notify subscribers
        self._notify_subscribers('created', product_data)
        
        product = self._dict_to_product(product_data)
        return product_service_pb2.CreateProductResponse(
            product=product,
            success=True
        )
    
    def UpdateProduct(self, request, context):
        """Update existing product"""
        logger.info(f"UpdateProduct called for ID: {request.id}")
        
        with self.lock:
            if request.id not in self.products_db:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f'Product with ID {request.id} not found')
                return product_service_pb2.UpdateProductResponse(
                    success=False,
                    error_message='Product not found'
                )
            
            # Update fields
            product_data = self.products_db[request.id]
            if request.name:
                product_data['name'] = request.name
            if request.description:
                product_data['description'] = request.description
            if request.price > 0:
                product_data['price'] = request.price
            if request.category_id > 0:
                product_data['category_id'] = request.category_id
            if request.stock >= 0:
                product_data['stock'] = request.stock
            
            product_data['updated_at'] = datetime.utcnow().isoformat() + 'Z'
        
        # Notify subscribers
        self._notify_subscribers('updated', product_data)
        
        product = self._dict_to_product(product_data)
        return product_service_pb2.UpdateProductResponse(
            product=product,
            success=True
        )
    
    def DeleteProduct(self, request, context):
        """Delete product"""
        logger.info(f"DeleteProduct called for ID: {request.id}")
        
        with self.lock:
            if request.id not in self.products_db:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f'Product with ID {request.id} not found')
                return product_service_pb2.DeleteProductResponse(
                    success=False,
                    error_message='Product not found'
                )
            
            product_data = self.products_db[request.id]
            del self.products_db[request.id]
        
        # Notify subscribers
        self._notify_subscribers('deleted', product_data)
        
        return product_service_pb2.DeleteProductResponse(success=True)
    
    def SubscribeToProductUpdates(self, request, context):
        """Server streaming: Subscribe to product updates"""
        logger.info(f"SubscribeToProductUpdates called for categories: {request.category_ids}")
        
        # Add subscriber
        subscriber = {
            'context': context,
            'category_ids': list(request.category_ids) if request.category_ids else [],
            'queue': []
        }
        
        with self.lock:
            self.subscribers.append(subscriber)
        
        try:
            # Send initial products
            for product_data in self.products_db.values():
                if not request.category_ids or product_data['category_id'] in request.category_ids:
                    event = product_service_pb2.ProductUpdateEvent(
                        event_type='initial',
                        product=self._dict_to_product(product_data),
                        timestamp=datetime.utcnow().isoformat() + 'Z'
                    )
                    yield event
            
            # Send updates as they come
            while context.is_active():
                if subscriber['queue']:
                    event = subscriber['queue'].pop(0)
                    yield event
                else:
                    time.sleep(0.1)  # Avoid busy waiting
                    
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
        finally:
            # Remove subscriber
            with self.lock:
                if subscriber in self.subscribers:
                    self.subscribers.remove(subscriber)
    
    def BatchCreateProducts(self, request_iterator, context):
        """Client streaming: Batch create products"""
        logger.info("BatchCreateProducts called")
        
        created_count = 0
        last_product = None
        
        for request in request_iterator:
            try:
                # Create product
                with self.lock:
                    product_data = {
                        'id': self.next_id,
                        'name': request.name,
                        'description': request.description,
                        'price': request.price,
                        'category_id': request.category_id,
                        'stock': request.stock,
                        'created_at': datetime.utcnow().isoformat() + 'Z',
                        'updated_at': datetime.utcnow().isoformat() + 'Z'
                    }
                    
                    self.products_db[self.next_id] = product_data
                    self.next_id += 1
                    created_count += 1
                    last_product = product_data
                
                # Notify subscribers
                self._notify_subscribers('created', product_data)
                
            except Exception as e:
                logger.error(f"Error creating product in batch: {e}")
        
        if last_product:
            return product_service_pb2.CreateProductResponse(
                product=self._dict_to_product(last_product),
                success=True
            )
        else:
            return product_service_pb2.CreateProductResponse(
                success=False,
                error_message='No products created'
            )
    
    def _dict_to_product(self, product_data: dict):
        """Convert dict to protobuf Product"""
        return product_service_pb2.Product(
            id=product_data['id'],
            name=product_data['name'],
            description=product_data['description'],
            price=product_data['price'],
            category_id=product_data['category_id'],
            stock=product_data['stock'],
            created_at=product_data['created_at'],
            updated_at=product_data['updated_at']
        )
    
    def _notify_subscribers(self, event_type: str, product_data: dict):
        """Notify all subscribers of product updates"""
        event = product_service_pb2.ProductUpdateEvent(
            event_type=event_type,
            product=self._dict_to_product(product_data),
            timestamp=datetime.utcnow().isoformat() + 'Z'
        )
        
        with self.lock:
            for subscriber in self.subscribers[:]:  # Copy list to avoid modification during iteration
                # Check if subscriber is interested in this category
                if not subscriber['category_ids'] or product_data['category_id'] in subscriber['category_ids']:
                    try:
                        if subscriber['context'].is_active():
                            subscriber['queue'].append(event)
                        else:
                            self.subscribers.remove(subscriber)
                    except Exception as e:
                        logger.error(f"Error notifying subscriber: {e}")
                        if subscriber in self.subscribers:
                            self.subscribers.remove(subscriber)

def serve():
    """Start gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    product_service_pb2_grpc.add_ProductServiceServicer_to_server(
        ProductServiceImpl(), server
    )
    
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    
    logger.info(f"Starting gRPC server on {listen_addr}")
    server.start()
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server")
        server.stop(0)

if __name__ == '__main__':
    serve()
```

### gRPC Client Implementation

```python
# grpc_client.py
import grpc
import logging
import threading
from typing import Iterator, List, Dict
import time

# Generated from protobuf
import product_service_pb2
import product_service_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductServiceClient:
    """gRPC client for Product Service"""
    
    def __init__(self, server_address: str = 'localhost:50051'):
        self.server_address = server_address
        self.channel = grpc.insecure_channel(server_address)
        self.stub = product_service_pb2_grpc.ProductServiceStub(self.channel)
    
    def get_product(self, product_id: int) -> dict:
        """Get single product"""
        try:
            request = product_service_pb2.GetProductRequest(id=product_id)
            response = self.stub.GetProduct(request)
            
            if response.found:
                return self._product_to_dict(response.product)
            else:
                return None
                
        except grpc.RpcError as e:
            logger.error(f"gRPC error getting product {product_id}: {e}")
            raise
    
   
                    max_price: float = 0, search: str = "", 
                    page: int = 1, page_size: int = 20) -> Dict:
        """Get products with filtering"""
        try:
            request = product_service_pb2.GetProductsRequest(
                category_id=category_id,
                min_price=min_price,
                max_price=max_price,
                search=search,
                page=page,
                page_size=page_size
            )
            
            response = self.stub.GetProducts(request)
            
            return {
                'products': [self._product_to_dict(p) for p in response.products],
                'total_count': response.total_count,
                'page': response.page,
                'page_size': response.page_size
            }
            
        except grpc.RpcError as e:
            logger.error(f"gRPC error getting products: {e}")
            raise
    
    def create_product(self, name: str, description: str, price: float,
                      category_id: int, stock: int) -> dict:
        """Create new product"""
        try:
            request = product_service_pb2.CreateProductRequest(
                name=name,
                description=description,
                price=price,
                category_id=category_id,
                stock=stock
            )
            
            response = self.stub.CreateProduct(request)
            
            if response.success:
                return self._product_to_dict(response.product)
            else:
                raise Exception(response.error_message)
                
        except grpc.RpcError as e:
            logger.error(f"gRPC error creating product: {e}")
            raise
    
    def update_product(self, product_id: int, **updates) -> dict:
        """Update existing product"""
        try:
            request = product_service_pb2.UpdateProductRequest(
                id=product_id,
                **updates
            )
            
            response = self.stub.UpdateProduct(request)
            
            if response.success:
                return self._product_to_dict(response.product)
            else:
                raise Exception(response.error_message)
                
        except grpc.RpcError as e:
            logger.error(f"gRPC error updating product {product_id}: {e}")
            raise
    
    def delete_product(self, product_id: int) -> bool:
        """Delete product"""
        try:
            request = product_service_pb2.DeleteProductRequest(id=product_id)
            response = self.stub.DeleteProduct(request)
            
            if not response.success:
                raise Exception(response.error_message)
            
            return True
            
        except grpc.RpcError as e:
            logger.error(f"gRPC error deleting product {product_id}: {e}")
            raise
    
    def subscribe_to_updates(self, category_ids: List[int] = None) -> Iterator:
        """Subscribe to product updates (server streaming)"""
        try:
            request = product_service_pb2.SubscribeToUpdatesRequest(
                category_ids=category_ids or []
            )
            
            for event in self.stub.SubscribeToProductUpdates(request):
                yield {
                    'event_type': event.event_type,
                    'product': self._product_to_dict(event.product),
                    'timestamp': event.timestamp
                }
                
        except grpc.RpcError as e:
            logger.error(f"gRPC error subscribing to updates: {e}")
            raise
    
    def batch_create_products(self, products: List[Dict]) -> dict:
        """Batch create products (client streaming)"""
        def request_generator():
            for product in products:
                yield product_service_pb2.CreateProductRequest(**product)
        
        try:
            response = self.stub.BatchCreateProducts(request_generator())
            
            if response.success:
                return self._product_to_dict(response.product)
            else:
                raise Exception(response.error_message)
                
        except grpc.RpcError as e:
            logger.error(f"gRPC error batch creating products: {e}")
            raise
    
    def _product_to_dict(self, product) -> dict:
        """Convert protobuf Product to dict"""
        return {
            'id': product.id,
            'name': product.name,
            'description': product.description,
            'price': product.price,
            'category_id': product.category_id,
            'stock': product.stock,
            'created_at': product.created_at,
            'updated_at': product.updated_at
        }
    
    def close(self):
        """Close gRPC channel"""
        self.channel.close()

# Example usage
if __name__ == '__main__':
    client = ProductServiceClient()
    
    try:
        # Get products
        products = client.get_products(category_id=1)
        print(f"Found {len(products['products'])} products")
        
        # Create product
        new_product = client.create_product(
            name='Test Product',
            description='A test product via gRPC',
            price=99.99,
            category_id=1,
            stock=10
        )
        print(f"Created product: {new_product['name']}")
        
        # Subscribe to updates in a separate thread
        def listen_for_updates():
            for event in client.subscribe_to_updates([1]):
                print(f"Received update: {event['event_type']} - {event['product']['name']}")
        
        update_thread = threading.Thread(target=listen_for_updates)
        update_thread.daemon = True
        update_thread.start()
        
        # Make some changes to trigger updates
        time.sleep(1)
        client.update_product(new_product['id'], name='Updated Test Product')
        
        time.sleep(2)
        client.delete_product(new_product['id'])
        
        time.sleep(1)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()
```

## Circuit Breaker Pattern

The Circuit Breaker pattern prevents cascading failures in distributed systems by monitoring service calls and "opening" when failures exceed a threshold, preventing further calls to the failing service.

### Circuit Breaker Implementation

```python
# circuit_breaker.py
import time
import threading
import logging
from enum import Enum
from typing import Callable, Any, Optional
from dataclasses import dataclass
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Blocking calls
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5      # Number of failures to open circuit
    recovery_timeout: int = 60      # Seconds before trying half-open
    success_threshold: int = 3      # Successes needed to close from half-open
    timeout: int = 10              # Request timeout
    expected_exceptions: tuple = (Exception,)

class CircuitBreaker:
    """Circuit breaker implementation for service calls"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} moved to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker {self.name} is OPEN"
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exceptions as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0

                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.name} CLOSED after recovery")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.success_count = 0
                logger.warning(f"Circuit breaker {self.name} reopened during recovery")
            elif (self.state == CircuitState.CLOSED and 
                  self.failure_count >= self.config.failure_threshold):
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker {self.name} OPENED after {self.failure_count} failures")
    
    def get_state(self) -> dict:
        """Get current circuit breaker state"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time
        }

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass

# HTTP client with circuit breaker
class ResilientHTTPClient:
    """HTTP client with circuit breaker, retry, and timeout"""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.default_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30,
            success_threshold=2,
            timeout=10,
            expected_exceptions=(requests.RequestException,)
        )
    
    def _get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service"""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(
                name=service_name,
                config=self.default_config
            )
        return self.circuit_breakers[service_name]
    
    def get(self, url: str, service_name: str = None, **kwargs) -> requests.Response:
        """GET request with circuit breaker"""
        return self._make_request('GET', url, service_name, **kwargs)
    
    def post(self, url: str, service_name: str = None, **kwargs) -> requests.Response:
        """POST request with circuit breaker"""
        return self._make_request('POST', url, service_name, **kwargs)
    
    def _make_request(self, method: str, url: str, service_name: str = None, **kwargs) -> requests.Response:
        """Make HTTP request with circuit breaker protection"""
        if not service_name:
            # Extract service name from URL
            from urllib.parse import urlparse
            parsed = urlparse(url)
            service_name = f"{parsed.hostname}:{parsed.port or 80}"
        
        circuit_breaker = self._get_circuit_breaker(service_name)
        
        def make_request():
            kwargs.setdefault('timeout', self.default_config.timeout)
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        
        return circuit_breaker.call(make_request)
    
    def get_all_states(self) -> dict:
        """Get states of all circuit breakers"""
        return {
            name: cb.get_state() 
            for name, cb in self.circuit_breakers.items()
        }

# Service client with circuit breaker
class ProductServiceClient:
    """Product service client with circuit breaker"""
    
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url
        self.http_client = ResilientHTTPClient()
        self.service_name = "product-service"
    
    def get_product(self, product_id: int) -> dict:
        """Get product with circuit breaker protection"""
        try:
            url = f"{self.base_url}/api/v1/products/{product_id}"
            response = self.http_client.get(url, self.service_name)
            return response.json()
        except CircuitBreakerOpenException:
            # Return fallback data when circuit is open
            return self._get_product_fallback(product_id)
        except requests.RequestException as e:
            logger.error(f"Product service request failed: {e}")
            return self._get_product_fallback(product_id)
    
    def get_products(self, **filters) -> dict:
        """Get products with circuit breaker protection"""
        try:
            url = f"{self.base_url}/api/v1/products"
            response = self.http_client.get(url, self.service_name, params=filters)
            return response.json()
        except CircuitBreakerOpenException:
            return self._get_products_fallback()
        except requests.RequestException as e:
            logger.error(f"Product service request failed: {e}")
            return self._get_products_fallback()
    
    def _get_product_fallback(self, product_id: int) -> dict:
        """Fallback response when product service is unavailable"""
        return {
            'data': {
                'id': product_id,
                'name': 'Product information temporarily unavailable',
                'price': 0.0,
                'stock': 0,
                'available': False
            },
            'source': 'fallback'
        }
    
    def _get_products_fallback(self) -> dict:
        """Fallback response when product service is unavailable"""
        return {
            'data': [],
            'message': 'Product catalog temporarily unavailable',
            'source': 'fallback'
        }
    
    def get_circuit_breaker_state(self) -> dict:
        """Get circuit breaker state for monitoring"""
        return self.http_client.circuit_breakers.get(self.service_name, {}).get_state()

# Usage example
if __name__ == '__main__':
    client = ProductServiceClient()
    
    # Test circuit breaker behavior
    for i in range(10):
        try:
            product = client.get_product(1)
            print(f"Attempt {i+1}: {product.get('source', 'service')}")
            
            # Check circuit breaker state
            state = client.get_circuit_breaker_state()
            print(f"Circuit breaker state: {state}")
            
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
        
        time.sleep(2)
```

## Best Practices

### 1. Design Principles

```python
# design_principles.py

# 1. Idempotency
class IdempotentOperations:
    """Ensure operations can be safely retried"""
    
    def __init__(self):
        self.processed_requests = set()
    
    def create_order(self, idempotency_key: str, order_data: dict) -> dict:
        """Create order with idempotency protection"""
        if idempotency_key in self.processed_requests:
            # Return cached result for duplicate request
            return self._get_cached_order(idempotency_key)
        
        # Process order
        order = self._process_order(order_data)
        
        # Cache result
        self.processed_requests.add(idempotency_key)
        self._cache_order(idempotency_key, order)
        
        return order

# 2. Graceful Degradation
class GracefulDegradation:
    """Handle service failures gracefully"""
    
    def get_product_with_recommendations(self, product_id: int) -> dict:
        """Get product with optional recommendations"""
        # Essential data - must succeed
        try:
            product = self.product_service.get_product(product_id)
        except Exception as e:
            raise Exception(f"Critical service failure: {e}")
        
        # Optional data - degrade gracefully
        recommendations = []
        try:
            recommendations = self.recommendation_service.get_recommendations(product_id)
        except Exception as e:
            logger.warning(f"Recommendation service failed: {e}")
            # Continue without recommendations
        
        return {
            'product': product,
            'recommendations': recommendations,
            'recommendations_available': len(recommendations) > 0
        }

# 3. Bulkhead Pattern
class BulkheadPattern:
    """Isolate critical resources"""
    
    def __init__(self):
        # Separate thread pools for different operations
        self.critical_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="critical")
        self.non_critical_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="non-critical")
    
    def process_critical_request(self, request):
        """Process critical request with dedicated resources"""
        return self.critical_executor.submit(self._handle_critical, request)
    
    def process_non_critical_request(self, request):
        """Process non-critical request with separate resources"""
        return self.non_critical_executor.submit(self._handle_non_critical, request)
```

### 2. Monitoring and Observability

```python
# monitoring.py
import time
import logging
from functools import wraps
from dataclasses import dataclass
from typing import Dict, Any
import json

@dataclass
class RequestMetrics:
    service_name: str
    endpoint: str
    method: str
    status_code: int
    duration_ms: float
    timestamp: str

class ServiceMonitoring:
    """Service monitoring and metrics collection"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.metrics = []
    
    def track_request(self, endpoint: str, method: str = 'GET'):
        """Decorator to track request metrics"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                status_code = 200
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    status_code = 500
                    raise
                finally:
                    duration = (time.time() - start_time) * 1000
                    
                    metric = RequestMetrics(
                        service_name=self.service_name,
                        endpoint=endpoint,
                        method=method,
                        status_code=status_code,
                        duration_ms=duration,
                        timestamp=time.time()
                    )
                    
                    self.metrics.append(metric)
                    self._log_metric(metric)
            
            return wrapper
        return decorator
    
    def _log_metric(self, metric: RequestMetrics):
        """Log metric for external collection"""
        logger.info(f"METRIC: {json.dumps(metric.__dict__)}")

# Distributed tracing
class DistributedTracing:
    """Simple distributed tracing implementation"""
    
    def __init__(self):
        self.traces = {}
    
    def start_span(self, operation_name: str, trace_id: str = None, parent_span_id: str = None):
        """Start a new span"""
        import uuid
        
        span_id = str(uuid.uuid4())
        if not trace_id:
            trace_id = span_id
        
        span = {
            'trace_id': trace_id,
            'span_id': span_id,
            'parent_span_id': parent_span_id,
            'operation_name': operation_name,
            'start_time': time.time(),
            'tags': {},
            'logs': []
        }
        
        self.traces[span_id] = span
        return span_id
    
    def finish_span(self, span_id: str, tags: Dict[str, Any] = None):
        """Finish span"""
        if span_id in self.traces:
            span = self.traces[span_id]
            span['end_time'] = time.time()
            span['duration'] = span['end_time'] - span['start_time']
            
            if tags:
                span['tags'].update(tags)
            
            self._export_span(span)
    
    def _export_span(self, span: dict):
        """Export span to tracing system"""
        logger.info(f"TRACE: {json.dumps(span)}")
```

### 3. Error Handling Strategies

```python
# error_handling.py
from enum import Enum
from typing import Optional, Any
from functools import wraps
import logging
import time

class ErrorType(Enum):
    VALIDATION_ERROR = "validation_error"
    BUSINESS_LOGIC_ERROR = "business_logic_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    INFRASTRUCTURE_ERROR = "infrastructure_error"

class ServiceError(Exception):
    """Base service error with context"""
    
    def __init__(self, message: str, error_type: ErrorType, 
                 error_code: str = None, details: dict = None):
        super().__init__(message)
        self.error_type = error_type
        self.error_code = error_code
        self.details = details or {}

class ErrorHandler:
    """Centralized error handling"""
    
    def handle_error(self, error: Exception, context: dict = None) -> dict:
        """Handle and format error response"""
        if isinstance(error, ServiceError):
            return self._handle_service_error(error, context)
        else:
            return self._handle_unexpected_error(error, context)
    
    def _handle_service_error(self, error: ServiceError, context: dict) -> dict:
        """Handle known service errors"""
        response = {
            'error': {
                'type': error.error_type.value,
                'message': str(error),
                'code': error.error_code,
                'details': error.details
            }
        }
        
        if context:
            response['error']['context'] = context
        
        # Log error
        logger.error(f"Service error: {error.error_type.value} - {error}")
        
        return response
    
    def _handle_unexpected_error(self, error: Exception, context: dict) -> dict:
        """Handle unexpected errors"""
        logger.exception("Unexpected error occurred")
        
        return {
            'error': {
                'type': ErrorType.INFRASTRUCTURE_ERROR.value,
                'message': 'An unexpected error occurred',
                'code': 'INTERNAL_ERROR'
            }
        }

# Retry strategies
class RetryStrategy:
    """Configurable retry strategies"""
    
    @staticmethod
    def exponential_backoff(attempt: int, base_delay: float = 1.0, 
                          max_delay: float = 60.0) -> float:
        """Calculate exponential backoff delay"""
        delay = base_delay * (2 ** attempt)
        return min(delay, max_delay)
    
    @staticmethod
    def linear_backoff(attempt: int, delay: float = 1.0) -> float:
        """Calculate linear backoff delay"""
        return delay * attempt
    
    @staticmethod
    def fixed_delay(attempt: int, delay: float = 1.0) -> float:
        """Fixed delay between retries"""
        return delay

def retry_with_backoff(max_retries: int = 3, strategy=RetryStrategy.exponential_backoff):
    """Decorator for retrying operations with backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        raise
                    
                    delay = strategy(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator
```

### Learning Objectives

By the end of this section, you should be able to:

- **Design and implement** synchronous and asynchronous communication patterns
- **Build RESTful APIs** following best practices with proper error handling and validation
- **Implement gRPC services** for high-performance communication with streaming support
- **Set up message-based communication** using RabbitMQ and Apache Kafka
- **Create API Gateway** with authentication, rate limiting, and request routing
- **Implement service discovery** using Consul for dynamic service registration
- **Apply circuit breaker pattern** to prevent cascading failures
- **Monitor and trace** distributed service communications
- **Handle errors gracefully** with proper fallback mechanisms

### Practical Exercises

**Exercise 1: Microservice Communication Chain**
```
Order Service → Product Service → Inventory Service → Payment Service
```
Implement this chain using both synchronous (REST/gRPC) and asynchronous (message queue) patterns.

**Exercise 2: API Gateway Implementation**
Create an API gateway that routes requests to multiple backend services with:
- JWT authentication
- Rate limiting per user
- Request/response transformation
- Circuit breaker for backend services

**Exercise 3: Event-Driven Architecture**
Implement an e-commerce system using event sourcing and CQRS patterns with:
- Order events (created, confirmed, shipped, delivered)
- Inventory events (reserved, released, updated)
- Payment events (processed, failed, refunded)

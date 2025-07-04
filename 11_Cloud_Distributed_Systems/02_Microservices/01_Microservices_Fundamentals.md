# Microservices Fundamentals

*Duration: 2 weeks*

## Table of Contents
- [What are Microservices?](#what-are-microservices)
- [Monolithic vs Microservices Architecture](#monolithic-vs-microservices-architecture)
- [Core Principles and Design Patterns](#core-principles-and-design-patterns)
- [Benefits of Microservices](#benefits-of-microservices)
- [Challenges and Trade-offs](#challenges-and-trade-offs)
- [When to Use Microservices](#when-to-use-microservices)
- [Practical Implementation Examples](#practical-implementation-examples)
- [Learning Objectives](#learning-objectives)
- [Study Materials](#study-materials)

## What are Microservices?

**Microservices** is an architectural style that structures an application as a collection of **loosely coupled services** that are:
- **Independently deployable**
- **Business capability focused**
- **Technology agnostic**
- **Fault tolerant**
- **Highly maintainable**

### Key Characteristics

```
┌─────────────────────────────────────────────────────────────┐
│                    Microservices Architecture               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   User      │  │  Product    │  │   Order     │         │
│  │  Service    │  │  Service    │  │  Service    │         │
│  │             │  │             │  │             │         │
│  │ Node.js     │  │   Java      │  │   Python    │         │
│  │ MongoDB     │  │ PostgreSQL  │  │   Redis     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                 │                 │              │
│         └─────────────────┼─────────────────┘              │
│                           │                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Notification│  │  Payment    │  │ Inventory   │         │
│  │  Service    │  │  Service    │  │  Service    │         │
│  │             │  │             │  │             │         │
│  │   Go        │  │    C#       │  │  Python     │         │
│  │  RabbitMQ   │  │ SQL Server  │  │ PostgreSQL  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

**Each microservice:**
- Owns its data and business logic
- Communicates via well-defined APIs (REST, gRPC, messaging)
- Can be developed by different teams
- Uses the most appropriate technology stack
- Scales independently based on demand

## Monolithic vs Microservices Architecture

### Monolithic Architecture

A **monolithic application** is deployed as a single unit where all components are interconnected and interdependent.

```python
# Example: Monolithic E-commerce Application
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = Flask(__name__)
Base = declarative_base()

# All models in one application
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50))
    email = Column(String(100))

class Product(Base):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    price = Column(Float)

class Order(Base):
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    product_id = Column(Integer)
    quantity = Column(Integer)

# All business logic in one application
@app.route('/users', methods=['POST'])
def create_user():
    # User management logic
    pass

@app.route('/products', methods=['GET'])
def get_products():
    # Product catalog logic
    pass

@app.route('/orders', methods=['POST'])
def create_order():
    # Order processing logic
    # Payment processing logic
    # Inventory management logic
    # Notification logic
    pass

@app.route('/payments', methods=['POST'])
def process_payment():
    # Payment processing logic
    pass

if __name__ == '__main__':
    app.run(debug=True)
```

**Monolithic Characteristics:**
- Single deployable unit
- Shared database
- Single technology stack
- Tight coupling between components
- All-or-nothing scaling

### Microservices Architecture

The same e-commerce application broken down into microservices:

#### 1. User Service
```python
# user_service.py
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import bcrypt
import jwt

app = Flask(__name__)
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True)
    email = Column(String(100), unique=True)
    password_hash = Column(String(255))

# Database setup
engine = create_engine('postgresql://user:pass@user-db:5432/userdb')
Session = sessionmaker(bind=engine)

@app.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    session = Session()
    
    # Hash password
    password_hash = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())
    
    user = User(
        username=data['username'],
        email=data['email'],
        password_hash=password_hash.decode('utf-8')
    )
    
    session.add(user)
    session.commit()
    
    return jsonify({
        'id': user.id,
        'username': user.username,
        'email': user.email
    }), 201

@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    session = Session()
    user = session.query(User).filter(User.id == user_id).first()
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'id': user.id,
        'username': user.username,
        'email': user.email
    })

@app.route('/users/authenticate', methods=['POST'])
def authenticate_user():
    data = request.get_json()
    session = Session()
    
    user = session.query(User).filter(User.username == data['username']).first()
    
    if user and bcrypt.checkpw(data['password'].encode('utf-8'), user.password_hash.encode('utf-8')):
        token = jwt.encode({'user_id': user.id}, 'secret_key', algorithm='HS256')
        return jsonify({'token': token})
    
    return jsonify({'error': 'Invalid credentials'}), 401

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
```

#### 2. Product Service
```python
# product_service.py
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import redis

app = Flask(__name__)
Base = declarative_base()
redis_client = redis.Redis(host='product-cache', port=6379, db=0)

class Product(Base):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    description = Column(Text)
    price = Column(Float)
    category_id = Column(Integer)

# Database setup
engine = create_engine('postgresql://user:pass@product-db:5432/productdb')
Session = sessionmaker(bind=engine)

@app.route('/products', methods=['GET'])
def get_products():
    # Check cache first
    cached_products = redis_client.get('products:all')
    if cached_products:
        return jsonify(eval(cached_products.decode('utf-8')))
    
    session = Session()
    products = session.query(Product).all()
    
    product_list = [{
        'id': p.id,
        'name': p.name,
        'description': p.description,
        'price': p.price,
        'category_id': p.category_id
    } for p in products]
    
    # Cache results for 5 minutes
    redis_client.setex('products:all', 300, str(product_list))
    
    return jsonify(product_list)

@app.route('/products/<int:product_id>', methods=['GET'])
def get_product(product_id):
    session = Session()
    product = session.query(Product).filter(Product.id == product_id).first()
    
    if not product:
        return jsonify({'error': 'Product not found'}), 404
    
    return jsonify({
        'id': product.id,
        'name': product.name,
        'description': product.description,
        'price': product.price,
        'category_id': product.category_id
    })

@app.route('/products', methods=['POST'])
def create_product():
    data = request.get_json()
    session = Session()
    
    product = Product(
        name=data['name'],
        description=data['description'],
        price=data['price'],
        category_id=data['category_id']
    )
    
    session.add(product)
    session.commit()
    
    # Invalidate cache
    redis_client.delete('products:all')
    
    return jsonify({
        'id': product.id,
        'name': product.name,
        'description': product.description,
        'price': product.price,
        'category_id': product.category_id
    }), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
```

#### 3. Order Service
```python
# order_service.py
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import requests
import pika
from datetime import datetime

app = Flask(__name__)
Base = declarative_base()

class Order(Base):
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    total_amount = Column(Float)
    status = Column(String(50), default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)

class OrderItem(Base):
    __tablename__ = 'order_items'
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer)
    product_id = Column(Integer)
    quantity = Column(Integer)
    price = Column(Float)

# Database setup
engine = create_engine('postgresql://user:pass@order-db:5432/orderdb')
Session = sessionmaker(bind=engine)

# RabbitMQ setup for messaging
def publish_event(event_type, data):
    connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
    channel = connection.channel()
    
    channel.exchange_declare(exchange='events', exchange_type='topic')
    
    channel.basic_publish(
        exchange='events',
        routing_key=event_type,
        body=str(data)
    )
    
    connection.close()

@app.route('/orders', methods=['POST'])
def create_order():
    data = request.get_json()
    session = Session()
    
    # Validate user exists (call User Service)
    user_response = requests.get(f'http://user-service:5001/users/{data["user_id"]}')
    if user_response.status_code != 200:
        return jsonify({'error': 'Invalid user'}), 400
    
    # Calculate total and validate products
    total_amount = 0
    validated_items = []
    
    for item in data['items']:
        # Get product details (call Product Service)
        product_response = requests.get(f'http://product-service:5002/products/{item["product_id"]}')
        if product_response.status_code != 200:
            return jsonify({'error': f'Invalid product {item["product_id"]}'}), 400
        
        product = product_response.json()
        item_total = product['price'] * item['quantity']
        total_amount += item_total
        
        validated_items.append({
            'product_id': item['product_id'],
            'quantity': item['quantity'],
            'price': product['price']
        })
    
    # Create order
    order = Order(
        user_id=data['user_id'],
        total_amount=total_amount
    )
    
    session.add(order)
    session.commit()
    
    # Create order items
    for item in validated_items:
        order_item = OrderItem(
            order_id=order.id,
            product_id=item['product_id'],
            quantity=item['quantity'],
            price=item['price']
        )
        session.add(order_item)
    
    session.commit()
    
    # Publish order created event
    publish_event('order.created', {
        'order_id': order.id,
        'user_id': order.user_id,
        'total_amount': order.total_amount,
        'items': validated_items
    })
    
    return jsonify({
        'id': order.id,
        'user_id': order.user_id,
        'total_amount': order.total_amount,
        'status': order.status,
        'created_at': order.created_at.isoformat()
    }), 201

@app.route('/orders/<int:order_id>', methods=['GET'])
def get_order(order_id):
    session = Session()
    order = session.query(Order).filter(Order.id == order_id).first()
    
    if not order:
        return jsonify({'error': 'Order not found'}), 404
    
    order_items = session.query(OrderItem).filter(OrderItem.order_id == order_id).all()
    
    return jsonify({
        'id': order.id,
        'user_id': order.user_id,
        'total_amount': order.total_amount,
        'status': order.status,
        'created_at': order.created_at.isoformat(),
        'items': [{
            'product_id': item.product_id,
            'quantity': item.quantity,
            'price': item.price
        } for item in order_items]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
```

#### 4. Payment Service
```python
# payment_service.py
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pika
import requests
from datetime import datetime
import uuid

app = Flask(__name__)
Base = declarative_base()

class Payment(Base):
    __tablename__ = 'payments'
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer)
    amount = Column(Float)
    payment_method = Column(String(50))
    transaction_id = Column(String(100))
    status = Column(String(50), default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)

# Database setup
engine = create_engine('postgresql://user:pass@payment-db:5432/paymentdb')
Session = sessionmaker(bind=engine)

def publish_event(event_type, data):
    connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
    channel = connection.channel()
    channel.exchange_declare(exchange='events', exchange_type='topic')
    channel.basic_publish(exchange='events', routing_key=event_type, body=str(data))
    connection.close()

def process_payment_gateway(amount, payment_method, card_details):
    """Simulate payment gateway processing"""
    # In real implementation, this would call Stripe, PayPal, etc.
    import random
    
    # Simulate processing time
    import time
    time.sleep(1)
    
    # Simulate 95% success rate
    if random.random() < 0.95:
        return {
            'success': True,
            'transaction_id': str(uuid.uuid4()),
            'message': 'Payment processed successfully'
        }
    else:
        return {
            'success': False,
            'transaction_id': None,
            'message': 'Payment failed: Insufficient funds'
        }

@app.route('/payments', methods=['POST'])
def process_payment():
    data = request.get_json()
    session = Session()
    
    # Validate order exists
    order_response = requests.get(f'http://order-service:5003/orders/{data["order_id"]}')
    if order_response.status_code != 200:
        return jsonify({'error': 'Invalid order'}), 400
    
    order = order_response.json()
    
    # Process payment through gateway
    gateway_result = process_payment_gateway(
        amount=order['total_amount'],
        payment_method=data['payment_method'],
        card_details=data.get('card_details', {})
    )
    
    # Create payment record
    payment = Payment(
        order_id=data['order_id'],
        amount=order['total_amount'],
        payment_method=data['payment_method'],
        transaction_id=gateway_result['transaction_id'],
        status='completed' if gateway_result['success'] else 'failed'
    )
    
    session.add(payment)
    session.commit()
    
    # Publish payment event
    event_type = 'payment.completed' if gateway_result['success'] else 'payment.failed'
    publish_event(event_type, {
        'payment_id': payment.id,
        'order_id': payment.order_id,
        'amount': payment.amount,
        'transaction_id': payment.transaction_id,
        'status': payment.status
    })
    
    return jsonify({
        'id': payment.id,
        'order_id': payment.order_id,
        'amount': payment.amount,
        'status': payment.status,
        'transaction_id': payment.transaction_id,
        'message': gateway_result['message']
    }), 201 if gateway_result['success'] else 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004)
```

#### 5. Notification Service
```python
# notification_service.py
from flask import Flask
import pika
import json
import requests
import smtplib
from email.mime.text import MIMEText

app = Flask(__name__)

def send_email(to_email, subject, body):
    """Send email notification"""
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'noreply@ecommerce.com'
    msg['To'] = to_email
    
    # In real implementation, configure SMTP server
    print(f"Email sent to {to_email}: {subject}")

def send_sms(phone_number, message):
    """Send SMS notification"""
    # In real implementation, integrate with SMS service
    print(f"SMS sent to {phone_number}: {message}")

def handle_order_created(data):
    """Handle order created event"""
    order_data = eval(data.decode('utf-8'))
    
    # Get user details
    user_response = requests.get(f'http://user-service:5001/users/{order_data["user_id"]}')
    if user_response.status_code == 200:
        user = user_response.json()
        
        send_email(
            user['email'],
            'Order Confirmation',
            f'Your order #{order_data["order_id"]} has been created. Total: ${order_data["total_amount"]}'
        )

def handle_payment_completed(data):
    """Handle payment completed event"""
    payment_data = eval(data.decode('utf-8'))
    
    # Get order details
    order_response = requests.get(f'http://order-service:5003/orders/{payment_data["order_id"]}')
    if order_response.status_code == 200:
        order = order_response.json()
        
        # Get user details
        user_response = requests.get(f'http://user-service:5001/users/{order["user_id"]}')
        if user_response.status_code == 200:
            user = user_response.json()
            
            send_email(
                user['email'],
                'Payment Confirmation',
                f'Payment for order #{order["id"]} has been processed successfully.'
            )

def callback(ch, method, properties, body):
    """Handle incoming events"""
    routing_key = method.routing_key
    
    if routing_key == 'order.created':
        handle_order_created(body)
    elif routing_key == 'payment.completed':
        handle_payment_completed(body)
    elif routing_key == 'payment.failed':
        # Handle payment failure
        pass
    
    ch.basic_ack(delivery_tag=method.delivery_tag)

def start_consuming():
    """Start consuming events from RabbitMQ"""
    connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
    channel = connection.channel()
    
    channel.exchange_declare(exchange='events', exchange_type='topic')
    
    result = channel.queue_declare(queue='notification_queue', exclusive=True)
    queue_name = result.method.queue
    
    channel.queue_bind(exchange='events', queue=queue_name, routing_key='order.*')
    channel.queue_bind(exchange='events', queue=queue_name, routing_key='payment.*')
    
    channel.basic_consume(queue=queue_name, on_message_callback=callback)
    
    print('Notification service started. Waiting for events...')
    channel.start_consuming()

if __name__ == '__main__':
    start_consuming()
```

### Architecture Comparison

| Aspect | Monolithic | Microservices |
|--------|------------|---------------|
| **Deployment** | Single unit | Independent services |
| **Technology Stack** | Uniform | Diverse (polyglot) |
| **Database** | Shared | Per-service |
| **Scaling** | Scale entire app | Scale individual services |
| **Development Teams** | Single team | Multiple teams |
| **Testing** | Simpler integration | Complex integration |
| **Monitoring** | Single application | Distributed tracing |
| **Network Calls** | In-process | Inter-service calls |
| **Data Consistency** | ACID transactions | Eventual consistency |
| **Failure Impact** | Entire app down | Service-level failures |

## Core Principles and Design Patterns

### 1. Single Responsibility Principle
Each microservice should have **one reason to change** and should be responsible for a **single business capability**.

```python
# ❌ BAD: Service doing too many things
class UserOrderPaymentService:
    def create_user(self): pass
    def authenticate_user(self): pass
    def create_order(self): pass
    def process_payment(self): pass
    def send_notifications(self): pass

# ✅ GOOD: Separate services for separate responsibilities
class UserService:
    def create_user(self): pass
    def authenticate_user(self): pass

class OrderService:
    def create_order(self): pass
    def update_order_status(self): pass

class PaymentService:
    def process_payment(self): pass
    def refund_payment(self): pass
```

### 2. Domain-Driven Design (DDD)
Organize services around **business domains** rather than technical layers.

```
Business Domains → Microservices Mapping:

E-commerce Platform:
├── User Management Domain → User Service
├── Product Catalog Domain → Product Service
├── Order Management Domain → Order Service
├── Payment Processing Domain → Payment Service
├── Inventory Management Domain → Inventory Service
├── Shipping & Logistics Domain → Shipping Service
└── Customer Support Domain → Support Service
```

### 3. Database per Service
Each microservice owns its data and database schema.

```yaml
# docker-compose.yml - Database isolation
version: '3.8'
services:
  user-service:
    build: ./user-service
    depends_on:
      - user-db
    environment:
      - DATABASE_URL=postgresql://user:pass@user-db:5432/userdb

  user-db:
    image: postgres:13
    environment:
      POSTGRES_DB: userdb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass

  product-service:
    build: ./product-service
    depends_on:
      - product-db
    environment:
      - DATABASE_URL=mongodb://product-db:27017/productdb

  product-db:
    image: mongo:4.4

  order-service:
    build: ./order-service
    depends_on:
      - order-db
    environment:
      - DATABASE_URL=postgresql://user:pass@order-db:5432/orderdb

  order-db:
    image: postgres:13
    environment:
      POSTGRES_DB: orderdb
```

### 4. API Gateway Pattern
Centralized entry point for client requests with routing, authentication, and cross-cutting concerns.

```python
# api_gateway.py using Flask and requests
from flask import Flask, request, jsonify
import requests
import jwt
from functools import wraps

app = Flask(__name__)

# Service registry
SERVICES = {
    'user': 'http://user-service:5001',
    'product': 'http://product-service:5002',
    'order': 'http://order-service:5003',
    'payment': 'http://payment-service:5004'
}

def authenticate(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            # Remove 'Bearer ' prefix
            token = token.split(' ')[1]
            jwt.decode(token, 'secret_key', algorithms=['HS256'])
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

def rate_limit():
    """Implement rate limiting logic"""
    # Redis-based rate limiting
    pass

def log_request():
    """Log all requests for monitoring"""
    print(f"{request.method} {request.path} from {request.remote_addr}")

@app.before_request
def before_request():
    log_request()
    # rate_limit()  # Uncomment to enable rate limiting

# User service routes
@app.route('/api/users', methods=['POST'])
def create_user():
    response = requests.post(f"{SERVICES['user']}/users", json=request.get_json())
    return jsonify(response.json()), response.status_code

@app.route('/api/users/<user_id>', methods=['GET'])
@authenticate
def get_user(user_id):
    response = requests.get(f"{SERVICES['user']}/users/{user_id}")
    return jsonify(response.json()), response.status_code

# Product service routes
@app.route('/api/products', methods=['GET'])
def get_products():
    response = requests.get(f"{SERVICES['product']}/products")
    return jsonify(response.json()), response.status_code

@app.route('/api/products/<product_id>', methods=['GET'])
def get_product(product_id):
    response = requests.get(f"{SERVICES['product']}/products/{product_id}")
    return jsonify(response.json()), response.status_code

# Order service routes
@app.route('/api/orders', methods=['POST'])
@authenticate
def create_order():
    response = requests.post(f"{SERVICES['order']}/orders", json=request.get_json())
    return jsonify(response.json()), response.status_code

@app.route('/api/orders/<order_id>', methods=['GET'])
@authenticate
def get_order(order_id):
    response = requests.get(f"{SERVICES['order']}/orders/{order_id}")
    return jsonify(response.json()), response.status_code

# Payment service routes
@app.route('/api/payments', methods=['POST'])
@authenticate
def process_payment():
    response = requests.post(f"{SERVICES['payment']}/payments", json=request.get_json())
    return jsonify(response.json()), response.status_code

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    service_health = {}
    
    for service_name, service_url in SERVICES.items():
        try:
            response = requests.get(f"{service_url}/health", timeout=5)
            service_health[service_name] = 'healthy' if response.status_code == 200 else 'unhealthy'
        except requests.RequestException:
            service_health[service_name] = 'unhealthy'
    
    overall_health = 'healthy' if all(status == 'healthy' for status in service_health.values()) else 'degraded'
    
    return jsonify({
        'status': overall_health,
        'services': service_health
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### 5. Event-Driven Architecture
Services communicate through events for loose coupling.

```python
# event_bus.py - Simple event bus implementation
import pika
import json
from typing import Dict, List, Callable

class EventBus:
    def __init__(self, rabbitmq_url='amqp://guest:guest@localhost:5672/'):
        self.connection = pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
        self.channel = self.connection.channel()
        self.channel.exchange_declare(exchange='events', exchange_type='topic')
        self.handlers: Dict[str, List[Callable]] = {}

    def publish(self, event_type: str, data: dict):
        """Publish an event"""
        self.channel.basic_publish(
            exchange='events',
            routing_key=event_type,
            body=json.dumps(data),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
                content_type='application/json'
            )
        )
        print(f"Published event: {event_type}")

    def subscribe(self, event_pattern: str, handler: Callable):
        """Subscribe to events matching pattern"""
        if event_pattern not in self.handlers:
            self.handlers[event_pattern] = []
        self.handlers[event_pattern].append(handler)

    def start_consuming(self, queue_name: str):
        """Start consuming events"""
        result = self.channel.queue_declare(queue=queue_name, durable=True)
        
        for pattern in self.handlers.keys():
            self.channel.queue_bind(
                exchange='events',
                queue=queue_name,
                routing_key=pattern
            )
        
        def callback(ch, method, properties, body):
            try:
                data = json.loads(body.decode('utf-8'))
                event_type = method.routing_key
                
                # Find matching handlers
                for pattern, handlers in self.handlers.items():
                    if self._pattern_matches(pattern, event_type):
                        for handler in handlers:
                            handler(event_type, data)
                
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                print(f"Error processing event: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        self.channel.basic_consume(queue=queue_name, on_message_callback=callback)
        self.channel.start_consuming()

    def _pattern_matches(self, pattern: str, event_type: str) -> bool:
        """Simple pattern matching (supports * wildcard)"""
        return pattern == event_type or pattern.endswith('*') and event_type.startswith(pattern[:-1])

# Usage example
if __name__ == '__main__':
    event_bus = EventBus()
    
    # Define event handlers
    def handle_order_events(event_type: str, data: dict):
        print(f"Order event received: {event_type} - {data}")
    
    def handle_payment_events(event_type: str, data: dict):
        print(f"Payment event received: {event_type} - {data}")
    
    # Subscribe to events
    event_bus.subscribe('order.*', handle_order_events)
    event_bus.subscribe('payment.*', handle_payment_events)
    
    # Start consuming
    event_bus.start_consuming('my_service_queue')
```

### 6. Circuit Breaker Pattern
Prevent cascading failures when services are unavailable.

```python
# circuit_breaker.py
import time
import requests
from enum import Enum
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "CLOSED"       # Normal operation
    OPEN = "OPEN"           # Failing, requests blocked
    HALF_OPEN = "HALF_OPEN" # Testing if service recovered

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60, expected_exception=requests.RequestException):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        return (time.time() - self.last_failure_time) >= self.timeout

    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Usage example
class OrderService:
    def __init__(self):
        self.payment_circuit = CircuitBreaker(failure_threshold=3, timeout=30)

    def process_order(self, order_data):
        try:
            # Call payment service with circuit breaker protection
            payment_result = self.payment_circuit.call(
                self._call_payment_service,
                order_data['payment_info']
            )
            return {'status': 'success', 'payment': payment_result}
        
        except Exception as e:
            # Fallback: queue order for later processing
            self._queue_order_for_retry(order_data)
            return {'status': 'queued', 'message': 'Payment service unavailable, order queued for processing'}

    def _call_payment_service(self, payment_info):
        response = requests.post(
            'http://payment-service:5004/payments',
            json=payment_info,
            timeout=5
        )
        response.raise_for_status()
        return response.json()

    def _queue_order_for_retry(self, order_data):
        # Queue order in Redis or message broker for retry
        print(f"Order {order_data['id']} queued for retry")
```

## Benefits of Microservices

### 1. **Independent Scaling**
Scale services based on individual demand patterns.

```yaml
# kubernetes-scaling.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: product-service
spec:
  replicas: 5  # Scale product service to 5 instances (high read load)
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: payment-service
spec:
  replicas: 3  # Scale payment service to 3 instances (medium load)
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-service
spec:
  replicas: 2  # Scale user service to 2 instances (low load)
```

**Horizontal Pod Autoscaler Example:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: product-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: product-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 2. **Technology Diversity (Polyglot Architecture)**
Choose the best technology for each service's requirements.

```python
# Performance comparison example

# User Service (Python + PostgreSQL) - Good for CRUD operations
class UserService:
    def __init__(self):
        self.db = PostgreSQLConnection()  # Strong consistency, ACID
    
    def create_user(self, user_data):
        # Python is great for rapid development
        return self.db.insert('users', user_data)

# Real-time Chat Service (Node.js + Redis) - High concurrency
const chatService = {
    redis: new Redis(),
    
    sendMessage: async (roomId, message) => {
        // Node.js excels at I/O operations
        await redis.publish(`room:${roomId}`, JSON.stringify(message));
    }
};

# Analytics Service (Go + ClickHouse) - High performance processing
package main

import (
    "database/sql"
    _ "github.com/mailru/go-clickhouse"
)

type AnalyticsService struct {
    db *sql.DB
}

func (as *AnalyticsService) ProcessEvents(events []Event) error {
    // Go provides excellent performance for data processing
    // ClickHouse optimized for analytical queries
    return as.db.Exec("INSERT INTO events VALUES (?)", events)
}

# Machine Learning Service (Python + TensorFlow) - ML capabilities
import tensorflow as tf
import numpy as np

class RecommendationService:
    def __init__(self):
        self.model = tf.keras.models.load_model('recommendation_model.h5')
    
    def get_recommendations(self, user_id, product_ids):
        # Python ecosystem is rich for ML/AI
        features = self.prepare_features(user_id, product_ids)
        predictions = self.model.predict(features)
        return self.format_recommendations(predictions)
```

### 3. **Fault Isolation**
Failures in one service don't bring down the entire system.

```python
# fault_tolerance_example.py
import requests
from typing import Optional

class EcommerceService:
    def __init__(self):
        self.services = {
            'user': 'http://user-service:5001',
            'product': 'http://product-service:5002',
            'inventory': 'http://inventory-service:5005',
            'recommendation': 'http://recommendation-service:5006'
        }

    def get_product_page(self, product_id: str, user_id: Optional[str] = None):
        """Get product page with graceful degradation"""
        result = {'product_id': product_id}
        
        # Core product info (critical)
        try:
            product_response = requests.get(f"{self.services['product']}/products/{product_id}", timeout=2)
            if product_response.status_code == 200:
                result['product'] = product_response.json()
            else:
                return {'error': 'Product not found'}, 404
        except requests.RequestException:
            return {'error': 'Product service unavailable'}, 503
        
        # Inventory info (important but not critical)
        try:
            inventory_response = requests.get(f"{self.services['inventory']}/inventory/{product_id}", timeout=1)
            if inventory_response.status_code == 200:
                result['inventory'] = inventory_response.json()
            else:
                result['inventory'] = {'status': 'unknown', 'quantity': 'Check availability'}
        except requests.RequestException:
            result['inventory'] = {'status': 'unavailable', 'message': 'Inventory service down'}
        
        # User-specific info (optional)
        if user_id:
            try:
                user_response = requests.get(f"{self.services['user']}/users/{user_id}", timeout=1)
                if user_response.status_code == 200:
                    result['user_info'] = user_response.json()
            except requests.RequestException:
                # Fail silently - user info is not critical
                pass
        
        # Recommendations (nice to have)
        try:
            rec_response = requests.get(
                f"{self.services['recommendation']}/recommendations/{product_id}?user_id={user_id}",
                timeout=0.5  # Very short timeout for recommendations
            )
            if rec_response.status_code == 200:
                result['recommendations'] = rec_response.json()
        except requests.RequestException:
            # Fail silently - recommendations are optional
            result['recommendations'] = []
        
        return result, 200

# Example: Even if recommendation and inventory services are down,
# the product page still loads with core product information
```

### 4. **Team Independence**
Different teams can work on different services independently.

```
Team Organization Structure:

Frontend Team
├── Web Application (React)
├── Mobile App (React Native)
└── API Gateway Management

User Management Team
├── User Service (Python/Django)
├── Authentication Service (Node.js)
└── User Database (PostgreSQL)

Product Team
├── Product Catalog Service (Java/Spring)
├── Search Service (Elasticsearch)
├── Product Database (MongoDB)
└── CDN Management

Order & Payment Team
├── Order Service (Python/Flask)
├── Payment Service (Go)
├── Order Database (PostgreSQL)
└── Payment Database (PostgreSQL)

Analytics Team
├── Analytics Service (Python/Pandas)
├── Recommendation Engine (Python/TensorFlow)
├── Data Pipeline (Apache Kafka)
└── Data Warehouse (ClickHouse)

DevOps Team
├── Container Orchestration (Kubernetes)
├── CI/CD Pipelines (Jenkins/GitLab)
├── Monitoring (Prometheus/Grafana)
└── Infrastructure (AWS/GCP/Azure)
```

### 5. **Faster Time to Market**
Teams can deploy features independently without waiting for other teams.

```yaml
# .github/workflows/microservice-ci-cd.yml
name: Microservice CI/CD

on:
  push:
    paths:
      - 'services/user-service/**'  # Only trigger for user service changes

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Unit Tests
        run: |
          cd services/user-service
          python -m pytest tests/

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker Image
        run: |
          cd services/user-service
          docker build -t user-service:${{ github.sha }} .
      
      - name: Deploy to Staging
        run: |
          kubectl set image deployment/user-service user-service=user-service:${{ github.sha }} -n staging
      
      - name: Run Integration Tests
        run: |
          cd services/user-service
          python -m pytest tests/integration/
      
      - name: Deploy to Production
        if: github.ref == 'refs/heads/main'
        run: |
          kubectl set image deployment/user-service user-service=user-service:${{ github.sha }} -n production
```

**Blue-Green Deployment Example:**
```bash
#!/bin/bash
# deploy.sh - Zero-downtime deployment

SERVICE_NAME="user-service"
NEW_VERSION=$1

# Deploy to blue environment
kubectl apply -f k8s/blue-deployment.yaml
kubectl set image deployment/${SERVICE_NAME}-blue ${SERVICE_NAME}=${SERVICE_NAME}:${NEW_VERSION} -n production

# Wait for deployment to be ready
kubectl rollout status deployment/${SERVICE_NAME}-blue -n production

# Run health checks
if curl -f http://user-service-blue.production.svc.cluster.local:5001/health; then
    echo "Health check passed, switching traffic"
    
    # Switch service to point to blue deployment
    kubectl patch service ${SERVICE_NAME} -p '{"spec":{"selector":{"version":"blue"}}}' -n production
    
    # Clean up old green deployment
    kubectl delete deployment ${SERVICE_NAME}-green -n production
    
    echo "Deployment completed successfully"
else
    echo "Health check failed, rolling back"
    kubectl delete deployment ${SERVICE_NAME}-blue -n production
    exit 1
fi
```

## Practical Implementation Examples

### Complete E-commerce Microservices Setup

#### Docker Compose Configuration
```yaml
# docker-compose.yml
version: '3.8'

services:
  # Infrastructure Services
  consul:
    image: consul:latest
    ports:
      - "8500:8500"
    command: agent -dev -client=0.0.0.0

  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: admin

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  # Databases
  user-db:
    image: postgres:13
    environment:
      POSTGRES_DB: userdb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - user_data:/var/lib/postgresql/data

  product-db:
    image: mongo:4.4
    environment:
      MONGO_INITDB_DATABASE: productdb
    volumes:
      - product_data:/data/db

  order-db:
    image: postgres:13
    environment:
      POSTGRES_DB: orderdb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - order_data:/var/lib/postgresql/data

  # Application Services
  api-gateway:
    build: ./api-gateway
    ports:
      - "8080:8080"
    depends_on:
      - consul
      - user-service
      - product-service
      - order-service
    environment:
      - CONSUL_HOST=consul

  user-service:
    build: ./user-service
    ports:
      - "5001:5001"
    depends_on:
      - user-db
      - consul
    environment:
      - DATABASE_URL=postgresql://user:password@user-db:5432/userdb
      - CONSUL_HOST=consul

  product-service:
    build: ./product-service
    ports:
      - "5002:5002"
    depends_on:
      - product-db
      - redis
      - consul
    environment:
      - DATABASE_URL=mongodb://product-db:27017/productdb
      - REDIS_URL=redis://redis:6379
      - CONSUL_HOST=consul

  order-service:
    build: ./order-service
    ports:
      - "5003:5003"
    depends_on:
      - order-db
      - rabbitmq
      - consul
    environment:
      - DATABASE_URL=postgresql://user:password@order-db:5432/orderdb
      - RABBITMQ_URL=amqp://admin:admin@rabbitmq:5672/
      - CONSUL_HOST=consul

  payment-service:
    build: ./payment-service
    ports:
      - "5004:5004"
    depends_on:
      - rabbitmq
      - consul
    environment:
      - RABBITMQ_URL=amqp://admin:admin@rabbitmq:5672/
      - CONSUL_HOST=consul

  notification-service:
    build: ./notification-service
    depends_on:
      - rabbitmq
    environment:
      - RABBITMQ_URL=amqp://admin:admin@rabbitmq:5672/

volumes:
  user_data:
  product_data:
  order_data:
```

#### Kubernetes Deployment
```yaml
# k8s/user-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-service
  labels:
    app: user-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: user-service
  template:
    metadata:
      labels:
        app: user-service
    spec:
      containers:
      - name: user-service
        image: user-service:latest
        ports:
        - containerPort: 5001
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: user-service-secret
              key: database-url
        livenessProbe:
          httpGet:
            path: /health
            port: 5001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 5001
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
---
apiVersion: v1
kind: Service
metadata:
  name: user-service
spec:
  selector:
    app: user-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5001
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: user-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: user-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

#### Monitoring and Observability Stack
```yaml
# monitoring/prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    
    scrape_configs:
    - job_name: 'user-service'
      static_configs:
      - targets: ['user-service:5001']
      metrics_path: /metrics
    
    - job_name: 'product-service'
      static_configs:
      - targets: ['product-service:5002']
      metrics_path: /metrics
    
    - job_name: 'order-service'
      static_configs:
      - targets: ['order-service:5003']
      metrics_path: /metrics
    
    - job_name: 'api-gateway'
      static_configs:
      - targets: ['api-gateway:8080']
      metrics_path: /metrics
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
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
        - name: config-volume
          mountPath: /etc/prometheus
        command:
        - '/bin/prometheus'
        - '--config.file=/etc/prometheus/prometheus.yml'
        - '--storage.tsdb.path=/prometheus'
      volumes:
      - name: config-volume
        configMap:
          name: prometheus-config
```

#### CI/CD Pipeline
```yaml
# .github/workflows/microservices-pipeline.yml
name: Microservices CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      user-service: ${{ steps.changes.outputs.user-service }}
      product-service: ${{ steps.changes.outputs.product-service }}
      order-service: ${{ steps.changes.outputs.order-service }}
      payment-service: ${{ steps.changes.outputs.payment-service }}
    steps:
    - uses: actions/checkout@v2
    - uses: dorny/paths-filter@v2
      id: changes
      with:
        filters: |
          user-service:
            - 'services/user-service/**'
          product-service:
            - 'services/product-service/**'
          order-service:
            - 'services/order-service/**'
          payment-service:
            - 'services/payment-service/**'

  test-user-service:
    needs: detect-changes
    if: ${{ needs.detect-changes.outputs.user-service == 'true' }}
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        cd services/user-service
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: |
        cd services/user-service
        pytest tests/ --cov=app --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v1

  build-and-deploy-user-service:
    needs: [detect-changes, test-user-service]
    if: ${{ needs.detect-changes.outputs.user-service == 'true' }}
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Build Docker image
      run: |
        cd services/user-service
        docker build -t user-service:${{ github.sha }} .
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker tag user-service:${{ github.sha }} ${{ secrets.DOCKER_REGISTRY }}/user-service:${{ github.sha }}
        docker push ${{ secrets.DOCKER_REGISTRY }}/user-service:${{ github.sha }}
    - name: Deploy to staging
      run: |
        kubectl set image deployment/user-service user-service=user-service:${{ secrets.DOCKER_REGISTRY }}/user-service:${{ github.sha }} -n staging
    - name: Run integration tests
      run: |
        cd services/user-service
        python -m pytest tests/integration/ --base-url=https://staging.api.example.com
    - name: Deploy to production
      if: github.ref == 'refs/heads/main'
      run: |
        kubectl set image deployment/user-service user-service=user-service:${{ secrets.DOCKER_REGISTRY }}/user-service:${{ github.sha }} -n production

  # Repeat similar jobs for other services...
```

### Real-world Examples

#### Netflix Architecture
```
Netflix Microservices (Simplified):

API Gateway (Zuul)
├── User Service (Java/Spring Boot)
├── Content Service (Java/Spring Boot)
├── Recommendation Service (Python/TensorFlow)
├── Streaming Service (C++/Java)
├── Billing Service (Java/Spring Boot)
└── Analytics Service (Scala/Spark)

Supporting Infrastructure:
├── Service Discovery (Eureka)
├── Configuration (Archaius)
├── Circuit Breaker (Hystrix)
├── Load Balancer (Ribbon)
├── Monitoring (Atlas)
└── Distributed Tracing (Zipkin)
```

#### Uber Architecture
```
Uber Platform:

Mobile Apps → API Gateway (Go) → Services:
├── User Service (Go)
├── Driver Service (Go)
├── Trip Service (Java)
├── Pricing Service (Python)
├── Payment Service (Java)
├── Notification Service (Node.js)
└── Analytics Service (Scala)

Data Stores:
├── User Data (MySQL)
├── Geolocation (Redis)
├── Trip Data (Cassandra)
├── Analytics (Hadoop/Spark)
└── Real-time Data (Kafka)
```

#### Amazon Architecture
```
Amazon E-commerce:

Frontend → API Gateway → Services:
├── Customer Service
├── Product Catalog Service
├── Inventory Service
├── Order Service
├── Payment Service
├── Shipping Service
├── Recommendation Service
└── Review Service

Each service:
├── Has its own database
├── Scales independently
├── Uses different technology stacks
└── Deployed independently
```

## Learning Objectives

By the end of this section, you should be able to:

### Core Understanding
- **Define microservices architecture** and explain its key characteristics
- **Compare and contrast** monolithic vs microservices architectures with specific examples
- **Identify when to use** microservices vs when to avoid them based on organizational and technical factors

### Design and Implementation
- **Design a microservices system** using domain-driven design principles
- **Implement core patterns** like API Gateway, Service Discovery, and Circuit Breaker
- **Handle data consistency** in distributed systems using patterns like Saga
- **Design for failure** with resilience patterns and fault tolerance

### Operational Excellence
- **Set up monitoring and observability** for distributed systems
- **Implement distributed tracing** to track requests across services
- **Design CI/CD pipelines** for independent service deployment
- **Configure service mesh** for advanced networking and security

### Self-Assessment Checklist

Before proceeding to advanced topics, ensure you can:

□ **Architecture Design**
  - [ ] Break down a monolithic application into microservices
  - [ ] Design service boundaries using domain-driven design
  - [ ] Choose appropriate communication patterns (sync vs async)

□ **Implementation Skills**
  - [ ] Create a microservice with proper health checks and metrics
  - [ ] Implement service-to-service communication with error handling
  - [ ] Set up service discovery and configuration management

□ **Operational Capabilities**
  - [ ] Deploy microservices using containers and orchestration
  - [ ] Monitor distributed system performance and errors
  - [ ] Implement log aggregation and distributed tracing

□ **Problem Solving**
  - [ ] Debug issues in distributed systems
  - [ ] Handle partial failures and implement fallback mechanisms
  - [ ] Manage data consistency across service boundaries

### Practical Exercises

#### Exercise 1: Decompose a Monolith
```
Given: A monolithic blog platform with:
- User management
- Post creation/editing
- Comment system
- Search functionality
- Analytics
- Email notifications

Task: Design microservices architecture
- Identify service boundaries
- Define APIs between services
- Plan data migration strategy
- Design deployment strategy
```

#### Exercise 2: Implement Service Communication
```python
# TODO: Complete this distributed order processing system

class OrderService:
    def create_order(self, order_data):
        # 1. Validate user exists (call User Service)
        # 2. Check product availability (call Product Service)
        # 3. Calculate pricing (call Payment Service)
        # 4. Process payment (call Payment Service)
        # 5. Update inventory (call Inventory Service)
        # 6. Create shipment (call Shipping Service)
        # 7. Send notifications (publish event)
        
        # Handle failures at each step
        # Implement compensation if needed
        pass

# Implement with proper error handling, retries, and circuit breakers
```

#### Exercise 3: Design for Scale
```
Scenario: E-commerce platform expecting:
- 1M users
- 10k orders/hour peak
- 100k product catalog
- Global availability

Design:
- Service scaling strategy
- Database partitioning
- Caching strategy
- CDN usage
- Regional deployment
```

## Study Materials

### Essential Reading

**Books:**
- **"Building Microservices"** by Sam Newman (2nd Edition)
- **"Microservices Patterns"** by Chris Richardson
- **"Release It!"** by Michael Nygard (Stability Patterns)
- **"Designing Data-Intensive Applications"** by Martin Kleppmann

**Articles & Papers:**
- **Martin Fowler's Microservices articles** - martinfowler.com/microservices
- **"CAP Theorem"** by Eric Brewer
- **"Distributed Systems for Fun and Profit"** by Mikito Takada

### Video Resources

**Conference Talks:**
- **"Microservices"** by Martin Fowler (GOTO 2014)
- **"Mastering Chaos - A Netflix Guide to Microservices"** by Josh Evans
- **"Debugging Microservices in Production"** by various speakers

**Course Series:**
- **"Microservices with Spring Boot"** - Spring.io
- **"Building Microservices with Node.js"** - Node.js Foundation
- **"Kubernetes for Microservices"** - CNCF

### Hands-on Labs

#### Lab 1: Basic Microservices (Week 1)
```
Objective: Build a simple e-commerce system

Services to implement:
1. User Service (authentication, user management)
2. Product Service (catalog, search)
3. Order Service (order processing)
4. API Gateway (routing, authentication)

Technologies:
- Python/Flask or Node.js/Express
- PostgreSQL or MongoDB
- Docker
- Basic service-to-service HTTP calls
```

#### Lab 2: Event-Driven Architecture (Week 2)
```
Objective: Add async communication and resilience

Enhancements:
1. Add RabbitMQ or Apache Kafka
2. Implement event-driven notifications
3. Add circuit breakers
4. Implement retry logic with exponential backoff
5. Add health checks and metrics

Tools:
- Message broker (RabbitMQ/Kafka)
- Prometheus for metrics
- Custom circuit breaker or Hystrix
```

#### Lab 3: Production-Ready Setup (Week 3)
```
Objective: Deploy to Kubernetes with full observability

Components:
1. Kubernetes deployment manifests
2. Service mesh (Istio) or Ingress
3. Distributed tracing (Jaeger)
4. Log aggregation (ELK stack)
5. Monitoring dashboards (Grafana)
6. CI/CD pipeline (GitHub Actions/GitLab CI)
```

### Technology Stacks

#### Beginner Stack
```
Runtime: Docker + Docker Compose
Languages: Python/Flask, Node.js/Express
Databases: PostgreSQL, Redis
Communication: HTTP REST APIs
Monitoring: Basic health checks
```

#### Intermediate Stack
```
Orchestration: Kubernetes
Service Mesh: Istio or Linkerd
Message Broker: RabbitMQ or Apache Kafka
Service Discovery: Consul or Kubernetes DNS
Circuit Breaker: Custom implementation
Monitoring: Prometheus + Grafana
```

#### Advanced Stack
```
Container Platform: Kubernetes + Helm
Service Mesh: Istio with advanced policies
Event Streaming: Apache Kafka + Schema Registry
Distributed Tracing: Jaeger or Zipkin
Observability: Prometheus + Grafana + ELK Stack
CI/CD: GitLab CI or GitHub Actions
Security: Vault, OPA, cert-manager
```

### Assessment Questions

#### Conceptual Questions
1. **When would you choose microservices over a modular monolith?**
2. **How do you maintain data consistency across microservices?**
3. **What are the trade-offs between synchronous and asynchronous communication?**
4. **How do you handle versioning of microservice APIs?**
5. **What strategies exist for testing microservices?**

#### Technical Challenges
```python
# Challenge 1: Design a saga for this business process
def book_travel():
    """
    Business Process:
    1. Reserve flight
    2. Reserve hotel
    3. Reserve car rental
    4. Process payment
    5. Send confirmation
    
    Design compensating actions for each step
    """
    pass

# Challenge 2: Implement distributed rate limiting
class DistributedRateLimiter:
    """
    Implement rate limiting that works across multiple
    instances of a service using Redis
    """
    pass

# Challenge 3: Design service discovery
class ServiceRegistry:
    """
    Implement service registration and discovery
    with health checking and load balancing
    """
    pass
```

#### Architecture Scenarios
1. **Design Instagram's architecture** using microservices
2. **Plan migration strategy** from monolith to microservices for a banking system
3. **Design resilience strategy** for a payment processing system
4. **Plan capacity and scaling** for a global streaming service

### Next Steps

After mastering microservices fundamentals:

1. **Advanced Patterns**: CQRS, Event Sourcing, Distributed Caching
2. **Service Mesh**: Advanced Istio, Linkerd, or Consul Connect
3. **Serverless Microservices**: AWS Lambda, Google Cloud Functions
4. **Container Orchestration**: Advanced Kubernetes, Helm, Operators
5. **Observability**: Advanced monitoring, APM, distributed tracing

## Next Section
[Service Design Patterns](02_Service_Design_Patterns.md)

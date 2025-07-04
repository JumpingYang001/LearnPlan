# Service Design Patterns

*Duration: 2 weeks*

## Overview
Master Domain-Driven Design (DDD) concepts, bounded contexts, aggregates, service decomposition strategies, and implement service design patterns for building robust, scalable microservices architectures.

## Core Concepts

### Domain-Driven Design (DDD) Fundamentals

#### What is Domain-Driven Design?

Domain-Driven Design is a software development approach that focuses on understanding and modeling complex business domains through collaboration between technical and domain experts.

**Key Principles:**
- **Ubiquitous Language**: Common vocabulary shared between developers and domain experts
- **Bounded Context**: Clear boundaries around domain models
- **Domain Model**: Rich representation of business logic and rules
- **Strategic Design**: High-level architectural decisions
- **Tactical Design**: Implementation patterns and techniques

#### DDD Building Blocks

**1. Entities**
Objects with unique identity that persist over time.

```python
from datetime import datetime
from typing import List, Optional
import uuid

class Customer:
    """Entity: Has unique identity and lifecycle"""
    
    def __init__(self, customer_id: str, email: str, name: str):
        self.customer_id = customer_id  # Unique identifier
        self.email = email
        self.name = name
        self.created_at = datetime.utcnow()
        self.is_active = True
    
    def deactivate(self):
        """Business operation"""
        self.is_active = False
    
    def change_email(self, new_email: str):
        """Business operation with validation"""
        if not self._is_valid_email(new_email):
            raise ValueError("Invalid email format")
        self.email = new_email
    
    def _is_valid_email(self, email: str) -> bool:
        return "@" in email and "." in email
    
    def __eq__(self, other):
        if not isinstance(other, Customer):
            return False
        return self.customer_id == other.customer_id
```

**2. Value Objects**
Objects defined by their attributes, with no identity.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)  # Immutable
class Money:
    """Value Object: Defined by its attributes"""
    amount: float
    currency: str
    
    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("Amount cannot be negative")
        if not self.currency:
            raise ValueError("Currency is required")
    
    def add(self, other: 'Money') -> 'Money':
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(self.amount + other.amount, self.currency)
    
    def multiply(self, factor: float) -> 'Money':
        return Money(self.amount * factor, self.currency)

@dataclass(frozen=True)
class Address:
    """Value Object: Complete address"""
    street: str
    city: str
    state: str
    postal_code: str
    country: str
    
    def __post_init__(self):
        if not all([self.street, self.city, self.state, self.postal_code, self.country]):
            raise ValueError("All address fields are required")
```

**3. Aggregates**
Cluster of domain objects treated as a single unit for data changes.

```python
from typing import List, Optional
from enum import Enum

class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class OrderItem:
    def __init__(self, product_id: str, product_name: str, price: Money, quantity: int):
        self.product_id = product_id
        self.product_name = product_name
        self.price = price
        self.quantity = quantity
    
    def total_price(self) -> Money:
        return self.price.multiply(self.quantity)

class Order:
    """Aggregate Root: Controls access to order items"""
    
    def __init__(self, order_id: str, customer_id: str):
        self.order_id = order_id
        self.customer_id = customer_id
        self.status = OrderStatus.PENDING
        self.items: List[OrderItem] = []
        self.shipping_address: Optional[Address] = None
        self.created_at = datetime.utcnow()
    
    def add_item(self, product_id: str, product_name: str, price: Money, quantity: int):
        """Business rule: Can only add items to pending orders"""
        if self.status != OrderStatus.PENDING:
            raise ValueError("Cannot modify non-pending order")
        
        # Check if item already exists
        existing_item = self._find_item(product_id)
        if existing_item:
            existing_item.quantity += quantity
        else:
            self.items.append(OrderItem(product_id, product_name, price, quantity))
    
    def remove_item(self, product_id: str):
        """Business rule: Can only remove items from pending orders"""
        if self.status != OrderStatus.PENDING:
            raise ValueError("Cannot modify non-pending order")
        
        self.items = [item for item in self.items if item.product_id != product_id]
    
    def confirm_order(self, shipping_address: Address):
        """Business operation with invariants"""
        if self.status != OrderStatus.PENDING:
            raise ValueError("Order is not in pending state")
        
        if not self.items:
            raise ValueError("Cannot confirm order without items")
        
        self.shipping_address = shipping_address
        self.status = OrderStatus.CONFIRMED
    
    def calculate_total(self) -> Money:
        """Aggregate business logic"""
        if not self.items:
            return Money(0.0, "USD")
        
        total = self.items[0].total_price()
        for item in self.items[1:]:
            total = total.add(item.total_price())
        
        return total
    
    def _find_item(self, product_id: str) -> Optional[OrderItem]:
        return next((item for item in self.items if item.product_id == product_id), None)
```

**4. Domain Services**
Operations that don't naturally belong to any entity or value object.

```python
class PricingService:
    """Domain Service: Cross-aggregate business logic"""
    
    def __init__(self, discount_repository):
        self.discount_repository = discount_repository
    
    def calculate_order_total_with_discounts(self, order: Order, customer: Customer) -> Money:
        """Complex pricing logic that spans multiple aggregates"""
        base_total = order.calculate_total()
        
        # Apply customer-specific discounts
        if customer.is_premium_member():
            discount = self._get_premium_discount(customer)
            base_total = self._apply_discount(base_total, discount)
        
        # Apply order-level discounts
        order_discounts = self.discount_repository.get_applicable_discounts(order)
        for discount in order_discounts:
            base_total = self._apply_discount(base_total, discount)
        
        return base_total
    
    def _get_premium_discount(self, customer: Customer) -> float:
        # Premium members get 10% discount
        return 0.10
    
    def _apply_discount(self, amount: Money, discount_rate: float) -> Money:
        discount_amount = amount.multiply(discount_rate)
        return amount.add(discount_amount.multiply(-1))  # Subtract discount
```

### Bounded Contexts

#### Understanding Bounded Contexts

A bounded context is a boundary within which a particular domain model is defined and applicable. It provides a clear separation of concerns and prevents different models from interfering with each other.

#### Visual Representation

```
E-commerce System Bounded Contexts

┌─────────────────────────────────────────────────────────────────┐
│                        E-commerce Platform                       │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Sales Context │ Inventory Context│      Shipping Context       │
│                 │                 │                             │
│ • Customer      │ • Product       │ • Shipment                  │
│ • Order         │ • Stock         │ • Carrier                   │
│ • Payment       │ • Warehouse     │ • Tracking                  │
│ • Invoice       │ • Supplier      │ • Delivery                  │
│                 │                 │                             │
│ Domain: Sales   │ Domain: Inv Mgmt│ Domain: Logistics           │
└─────────────────┴─────────────────┴─────────────────────────────┘
         │                   │                        │
         │                   │                        │
         ▼                   ▼                        ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│  Sales Service  │ │Inventory Service│ │  Shipping Service   │
│                 │ │                 │ │                     │
│ • Order Mgmt    │ │ • Stock Check   │ │ • Route Planning    │
│ • Customer Mgmt │ │ • Restock Alert │ │ • Delivery Tracking │
│ • Billing       │ │ • Supplier Mgmt │ │ • Carrier Mgmt      │
└─────────────────┘ └─────────────────┘ └─────────────────────┘
```

#### Practical Implementation

```python
# Sales Bounded Context
class SalesOrder:
    """Order in Sales context - focused on customer and billing"""
    def __init__(self, order_id: str, customer_id: str):
        self.order_id = order_id
        self.customer_id = customer_id
        self.billing_address = None
        self.payment_method = None
        self.total_amount = Money(0, "USD")

# Inventory Bounded Context  
class InventoryOrder:
    """Order in Inventory context - focused on stock and fulfillment"""
    def __init__(self, order_id: str):
        self.order_id = order_id
        self.reserved_items = []
        self.warehouse_location = None
        self.fulfillment_status = "pending"

# Shipping Bounded Context
class ShippingOrder:
    """Order in Shipping context - focused on delivery"""
    def __init__(self, order_id: str):
        self.order_id = order_id
        self.shipping_address = None
        self.carrier = None
        self.tracking_number = None
        self.delivery_status = "not_shipped"

# Context Integration via Events
class OrderCreatedEvent:
    def __init__(self, order_id: str, customer_id: str, items: list):
        self.order_id = order_id
        self.customer_id = customer_id
        self.items = items
        self.timestamp = datetime.utcnow()

class SalesService:
    def __init__(self, event_publisher):
        self.event_publisher = event_publisher
    
    def create_order(self, customer_id: str, items: list) -> SalesOrder:
        order_id = str(uuid.uuid4())
        order = SalesOrder(order_id, customer_id)
        
        # Publish event for other contexts
        event = OrderCreatedEvent(order_id, customer_id, items)
        self.event_publisher.publish(event)
        
        return order

class InventoryService:
    def handle_order_created(self, event: OrderCreatedEvent):
        # Create inventory order and reserve stock
        inventory_order = InventoryOrder(event.order_id)
        self._reserve_stock(inventory_order, event.items)

class ShippingService:
    def handle_order_created(self, event: OrderCreatedEvent):
        # Create shipping order for later fulfillment
        shipping_order = ShippingOrder(event.order_id)
        self._prepare_for_shipping(shipping_order)
```

## Example Code
```python
# Example: Service Decomposition
# Product Service and Order Service communicate via REST
# Product Service
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/products')
def get_products():
    return jsonify([{"id": 1, "name": "Book"}])

if __name__ == '__main__':
    app.run(port=5001)
```

### Service Decomposition Strategies

#### 1. Decompose by Business Capability

Split services based on what the business does, not how it does it.

```python
# E-commerce Business Capabilities

# Customer Management Service
class CustomerService:
    """Handles everything related to customer lifecycle"""
    
    def __init__(self, customer_repository, notification_service):
        self.customer_repository = customer_repository
        self.notification_service = notification_service
    
    def register_customer(self, customer_data: dict) -> Customer:
        # Validate customer data
        if not self._is_valid_customer_data(customer_data):
            raise ValueError("Invalid customer data")
        
        customer = Customer(
            customer_id=str(uuid.uuid4()),
            email=customer_data['email'],
            name=customer_data['name']
        )
        
        self.customer_repository.save(customer)
        self.notification_service.send_welcome_email(customer.email)
        
        return customer
    
    def update_customer_profile(self, customer_id: str, updates: dict):
        customer = self.customer_repository.get_by_id(customer_id)
        if not customer:
            raise ValueError("Customer not found")
        
        # Apply updates
        for field, value in updates.items():
            if hasattr(customer, field):
                setattr(customer, field, value)
        
        self.customer_repository.save(customer)
    
    def _is_valid_customer_data(self, data: dict) -> bool:
        required_fields = ['email', 'name']
        return all(field in data for field in required_fields)

# Product Catalog Service
class ProductCatalogService:
    """Manages product information and catalog operations"""
    
    def __init__(self, product_repository, search_service):
        self.product_repository = product_repository
        self.search_service = search_service
    
    def add_product(self, product_data: dict) -> Product:
        product = Product(
            product_id=str(uuid.uuid4()),
            name=product_data['name'],
            description=product_data['description'],
            price=Money(product_data['price'], product_data['currency']),
            category=product_data['category']
        )
        
        self.product_repository.save(product)
        self.search_service.index_product(product)
        
        return product
    
    def search_products(self, query: str, filters: dict = None) -> List[Product]:
        return self.search_service.search(query, filters)
    
    def get_product_by_id(self, product_id: str) -> Optional[Product]:
        return self.product_repository.get_by_id(product_id)

# Order Management Service
class OrderManagementService:
    """Handles order lifecycle and business rules"""
    
    def __init__(self, order_repository, customer_service, inventory_service, payment_service):
        self.order_repository = order_repository
        self.customer_service = customer_service
        self.inventory_service = inventory_service
        self.payment_service = payment_service
    
    def create_order(self, customer_id: str, items: List[dict]) -> Order:
        # Validate customer exists
        customer = self.customer_service.get_customer(customer_id)
        if not customer:
            raise ValueError("Customer not found")
        
        # Check inventory availability
        for item in items:
            if not self.inventory_service.is_available(item['product_id'], item['quantity']):
                raise ValueError(f"Product {item['product_id']} not available")
        
        # Create order
        order = Order(str(uuid.uuid4()), customer_id)
        for item in items:
            product = self.product_catalog_service.get_product_by_id(item['product_id'])
            order.add_item(item['product_id'], product.name, product.price, item['quantity'])
        
        self.order_repository.save(order)
        return order
    
    def process_payment(self, order_id: str, payment_details: dict):
        order = self.order_repository.get_by_id(order_id)
        if not order:
            raise ValueError("Order not found")
        
        total_amount = order.calculate_total()
        payment_result = self.payment_service.process_payment(
            amount=total_amount,
            payment_method=payment_details['method'],
            card_details=payment_details.get('card_details')
        )
        
        if payment_result.success:
            order.status = OrderStatus.CONFIRMED
            self.order_repository.save(order)
        else:
            raise ValueError("Payment failed")
```

#### 2. Decompose by Data Ownership

Each service owns and manages its specific data domain.

```python
# Data Ownership Pattern

# User Service - owns user data
class UserService:
    def __init__(self, user_db):
        self.user_db = user_db  # Dedicated user database
    
    def create_user(self, user_data: dict) -> dict:
        user = {
            'user_id': str(uuid.uuid4()),
            'username': user_data['username'],
            'email': user_data['email'],
            'created_at': datetime.utcnow()
        }
        self.user_db.users.insert_one(user)
        return user
    
    def get_user(self, user_id: str) -> dict:
        return self.user_db.users.find_one({'user_id': user_id})

# Profile Service - owns profile data
class ProfileService:
    def __init__(self, profile_db):
        self.profile_db = profile_db  # Dedicated profile database
    
    def create_profile(self, user_id: str, profile_data: dict) -> dict:
        profile = {
            'user_id': user_id,
            'first_name': profile_data['first_name'],
            'last_name': profile_data['last_name'],
            'bio': profile_data.get('bio', ''),
            'avatar_url': profile_data.get('avatar_url', ''),
            'updated_at': datetime.utcnow()
        }
        self.profile_db.profiles.insert_one(profile)
        return profile
    
    def get_profile(self, user_id: str) -> dict:
        return self.profile_db.profiles.find_one({'user_id': user_id})

# Preferences Service - owns user preferences
class PreferencesService:
    def __init__(self, preferences_db):
        self.preferences_db = preferences_db
    
    def set_preferences(self, user_id: str, preferences: dict) -> dict:
        pref_doc = {
            'user_id': user_id,
            'theme': preferences.get('theme', 'light'),
            'notifications': preferences.get('notifications', True),
            'language': preferences.get('language', 'en'),
            'updated_at': datetime.utcnow()
        }
        
        self.preferences_db.preferences.replace_one(
            {'user_id': user_id}, 
            pref_doc, 
            upsert=True
        )
        return pref_doc
```

#### 3. Decompose by Use Case (Vertical Slicing)

Organize services around specific user journeys or use cases.

```python
# Use Case-Driven Decomposition

# User Registration Use Case Service
class UserRegistrationService:
    """Handles complete user registration flow"""
    
    def __init__(self, user_repo, email_service, validation_service):
        self.user_repo = user_repo
        self.email_service = email_service
        self.validation_service = validation_service
    
    def register_user(self, registration_data: dict) -> dict:
        # Step 1: Validate registration data
        validation_result = self.validation_service.validate_registration(registration_data)
        if not validation_result.is_valid:
            raise ValueError(f"Validation failed: {validation_result.errors}")
        
        # Step 2: Check if user already exists
        if self.user_repo.get_by_email(registration_data['email']):
            raise ValueError("User with this email already exists")
        
        # Step 3: Create user account
        user = self._create_user_account(registration_data)
        
        # Step 4: Send welcome email
        self.email_service.send_welcome_email(user['email'], user['username'])
        
        # Step 5: Create default preferences
        self._create_default_preferences(user['user_id'])
        
        return {
            'user_id': user['user_id'],
            'status': 'registered',
            'next_step': 'email_verification'
        }
    
    def _create_user_account(self, data: dict) -> dict:
        user = {
            'user_id': str(uuid.uuid4()),
            'username': data['username'],
            'email': data['email'],
            'password_hash': self._hash_password(data['password']),
            'status': 'pending_verification',
            'created_at': datetime.utcnow()
        }
        self.user_repo.save(user)
        return user

# Order Fulfillment Use Case Service
class OrderFulfillmentService:
    """Handles complete order fulfillment process"""
    
    def __init__(self, order_repo, inventory_service, shipping_service, notification_service):
        self.order_repo = order_repo
        self.inventory_service = inventory_service
        self.shipping_service = shipping_service
        self.notification_service = notification_service
    
    def fulfill_order(self, order_id: str) -> dict:
        # Step 1: Get order details
        order = self.order_repo.get_by_id(order_id)
        if not order or order.status != OrderStatus.CONFIRMED:
            raise ValueError("Order not ready for fulfillment")
        
        # Step 2: Reserve inventory
        reservation_result = self.inventory_service.reserve_items(order.items)
        if not reservation_result.success:
            self.notification_service.notify_inventory_shortage(order_id)
            raise ValueError("Inventory reservation failed")
        
        # Step 3: Create shipment
        shipment = self.shipping_service.create_shipment(
            order_id=order_id,
            items=order.items,
            destination=order.shipping_address
        )
        
        # Step 4: Update order status
        order.status = OrderStatus.SHIPPED
        order.tracking_number = shipment.tracking_number
        self.order_repo.save(order)
        
        # Step 5: Notify customer
        self.notification_service.notify_order_shipped(
            customer_email=order.customer_email,
            tracking_number=shipment.tracking_number
        )
        
        return {
            'order_id': order_id,
            'status': 'fulfilled',
            'tracking_number': shipment.tracking_number
        }
```

#### 4. Decompose by Scalability Requirements

Split services based on different performance and scaling needs.

```python
# High-Traffic Read Service
class ProductSearchService:
    """Optimized for high-volume read operations"""
    
    def __init__(self, search_engine, cache, read_replicas):
        self.search_engine = search_engine  # Elasticsearch
        self.cache = cache  # Redis
        self.read_replicas = read_replicas  # Multiple read-only databases
    
    def search_products(self, query: str, filters: dict = None) -> List[dict]:
        # Check cache first
        cache_key = self._generate_cache_key(query, filters)
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Search in search engine
        results = self.search_engine.search(
            query=query,
            filters=filters,
            limit=50
        )
        
        # Cache results for future requests
        self.cache.set(cache_key, results, ttl=300)  # 5 minutes
        
        return results
    
    def get_product_details(self, product_id: str) -> dict:
        # Use read replicas for product details
        replica = self._select_least_loaded_replica()
        return replica.get_product(product_id)

# Write-Heavy Service
class InventoryUpdateService:
    """Optimized for high-volume write operations"""
    
    def __init__(self, write_db, event_bus, batch_processor):
        self.write_db = write_db  # Write-optimized database
        self.event_bus = event_bus
        self.batch_processor = batch_processor
    
    def update_stock_level(self, product_id: str, quantity_change: int):
        # Batch writes for better performance
        self.batch_processor.add_update({
            'product_id': product_id,
            'quantity_change': quantity_change,
            'timestamp': datetime.utcnow()
        })
        
        # Publish event for eventual consistency
        self.event_bus.publish({
            'event_type': 'stock_updated',
            'product_id': product_id,
            'quantity_change': quantity_change
        })
    
    def process_batch_updates(self):
        """Process accumulated updates in batch"""
        updates = self.batch_processor.get_pending_updates()
        if updates:
            self.write_db.bulk_update_stock(updates)
            self.batch_processor.clear_pending()

# Real-time Service
class LiveOrderTrackingService:
    """Real-time order status and location tracking"""
    
    def __init__(self, websocket_manager, location_service):
        self.websocket_manager = websocket_manager
        self.location_service = location_service
    
    def track_order_real_time(self, order_id: str, customer_id: str):
        # Establish WebSocket connection for real-time updates
        connection = self.websocket_manager.create_connection(
            customer_id=customer_id,
            channel=f"order_tracking_{order_id}"
        )
        
        # Start real-time location tracking
        self.location_service.start_tracking(
            order_id=order_id,
            callback=lambda location: self._send_location_update(connection, location)
        )
    
    def _send_location_update(self, connection, location_data):
        update = {
            'type': 'location_update',
            'latitude': location_data['lat'],
            'longitude': location_data['lng'],
            'estimated_arrival': location_data['eta'],
            'timestamp': datetime.utcnow().isoformat()
        }
        self.websocket_manager.send(connection, update)
```

### Service Communication Patterns

#### 1. Synchronous Communication

**REST API Communication**
```python
import requests
from typing import Optional
import asyncio
import aiohttp

class CustomerServiceClient:
    """Client for synchronous communication with Customer Service"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
    
    def get_customer(self, customer_id: str) -> Optional[dict]:
        try:
            response = requests.get(
                f"{self.base_url}/customers/{customer_id}",
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching customer {customer_id}: {e}")
            return None
    
    def create_customer(self, customer_data: dict) -> Optional[dict]:
        try:
            response = requests.post(
                f"{self.base_url}/customers",
                json=customer_data,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error creating customer: {e}")
            return None

# Usage in Order Service
class OrderService:
    def __init__(self, customer_client: CustomerServiceClient):
        self.customer_client = customer_client
    
    def create_order(self, customer_id: str, items: list) -> dict:
        # Synchronous call to Customer Service
        customer = self.customer_client.get_customer(customer_id)
        if not customer:
            raise ValueError("Customer not found")
        
        # Create order with customer information
        order = {
            'order_id': str(uuid.uuid4()),
            'customer_id': customer_id,
            'customer_name': customer['name'],
            'items': items,
            'status': 'pending'
        }
        
        return order
```

**Async/Await Pattern for Better Performance**
```python
class AsyncCustomerServiceClient:
    """Async client for better performance"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    async def get_customer(self, customer_id: str) -> Optional[dict]:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.base_url}/customers/{customer_id}"
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
            except aiohttp.ClientError as e:
                print(f"Error: {e}")
                return None
    
    async def get_multiple_customers(self, customer_ids: list) -> dict:
        """Fetch multiple customers concurrently"""
        tasks = [self.get_customer(cid) for cid in customer_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            customer_ids[i]: result 
            for i, result in enumerate(results) 
            if not isinstance(result, Exception) and result is not None
        }
```

#### 2. Asynchronous Communication

**Event-Driven Architecture with Message Queues**
```python
import json
import pika
from abc import ABC, abstractmethod

class EventPublisher:
    """Publisher for domain events"""
    
    def __init__(self, rabbitmq_url: str):
        self.connection = pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
        self.channel = self.connection.channel()
    
    def publish_event(self, event_type: str, event_data: dict, routing_key: str = ""):
        event = {
            'event_type': event_type,
            'event_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'data': event_data
        }
        
        self.channel.basic_publish(
            exchange='domain_events',
            routing_key=routing_key,
            body=json.dumps(event),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
                content_type='application/json'
            )
        )

class EventHandler(ABC):
    """Base class for event handlers"""
    
    @abstractmethod
    def handle(self, event: dict):
        pass

class OrderCreatedHandler(EventHandler):
    """Handles order created events"""
    
    def __init__(self, inventory_service, shipping_service):
        self.inventory_service = inventory_service
        self.shipping_service = shipping_service
    
    def handle(self, event: dict):
        order_data = event['data']
        order_id = order_data['order_id']
        items = order_data['items']
        
        # Reserve inventory
        try:
            self.inventory_service.reserve_items(order_id, items)
        except Exception as e:
            print(f"Failed to reserve inventory for order {order_id}: {e}")
            # Publish compensation event
            return
        
        # Prepare shipping
        try:
            self.shipping_service.prepare_shipment(order_id, order_data['shipping_address'])
        except Exception as e:
            print(f"Failed to prepare shipment for order {order_id}: {e}")
            # Compensate inventory reservation
            self.inventory_service.release_reservation(order_id)

class EventConsumer:
    """Consumer for processing events"""
    
    def __init__(self, rabbitmq_url: str, handlers: dict):
        self.connection = pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
        self.channel = self.connection.channel()
        self.handlers = handlers  # Map of event_type -> handler
    
    def start_consuming(self, queue_name: str):
        self.channel.queue_declare(queue=queue_name, durable=True)
        self.channel.basic_consume(
            queue=queue_name,
            on_message_callback=self._process_message,
            auto_ack=False
        )
        
        print(f"Started consuming from {queue_name}")
        self.channel.start_consuming()
    
    def _process_message(self, channel, method, properties, body):
        try:
            event = json.loads(body)
            event_type = event['event_type']
            
            if event_type in self.handlers:
                self.handlers[event_type].handle(event)
                channel.basic_ack(delivery_tag=method.delivery_tag)
            else:
                print(f"No handler for event type: {event_type}")
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        
        except Exception as e:
            print(f"Error processing message: {e}")
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

# Usage Example
class OrderService:
    def __init__(self, event_publisher: EventPublisher):
        self.event_publisher = event_publisher
    
    def create_order(self, customer_id: str, items: list) -> dict:
        order = {
            'order_id': str(uuid.uuid4()),
            'customer_id': customer_id,
            'items': items,
            'status': 'created',
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Save order to database
        # ... save logic ...
        
        # Publish event asynchronously
        self.event_publisher.publish_event(
            event_type='order_created',
            event_data=order,
            routing_key='orders'
        )
        
        return order

# Event Consumer Setup
order_handler = OrderCreatedHandler(inventory_service, shipping_service)
handlers = {
    'order_created': order_handler
}

consumer = EventConsumer('amqp://localhost', handlers)
consumer.start_consuming('order_events')
```

## Common Service Design Patterns

### 1. Database per Service Pattern

Each microservice has its own private database to ensure loose coupling.

```python
# User Service with its own database
class UserService:
    def __init__(self):
        # Dedicated user database
        self.db = self._connect_to_user_db()
    
    def _connect_to_user_db(self):
        # MongoDB for user data (document-based)
        from pymongo import MongoClient
        return MongoClient('mongodb://user-db:27017/users')
    
    def create_user(self, user_data: dict) -> dict:
        user = {
            'user_id': str(uuid.uuid4()),
            'email': user_data['email'],
            'username': user_data['username'],
            'created_at': datetime.utcnow()
        }
        self.db.users.insert_one(user)
        return user

# Order Service with its own database
class OrderService:
    def __init__(self):
        # Dedicated order database
        self.db = self._connect_to_order_db()
    
    def _connect_to_order_db(self):
        # PostgreSQL for transactional order data
        import psycopg2
        return psycopg2.connect(
            host="order-db",
            database="orders",
            user="order_user",
            password="order_pass"
        )
    
    def create_order(self, order_data: dict) -> dict:
        cursor = self.db.cursor()
        order_id = str(uuid.uuid4())
        
        cursor.execute("""
            INSERT INTO orders (order_id, customer_id, total_amount, status, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            order_id,
            order_data['customer_id'],
            order_data['total_amount'],
            'pending',
            datetime.utcnow()
        ))
        
        self.db.commit()
        return {'order_id': order_id, 'status': 'created'}

# Analytics Service with its own database
class AnalyticsService:
    def __init__(self):
        # Time-series database for analytics
        self.db = self._connect_to_analytics_db()
    
    def _connect_to_analytics_db(self):
        # InfluxDB for time-series analytics data
        from influxdb import InfluxDBClient
        return InfluxDBClient(host='analytics-db', port=8086, database='analytics')
    
    def record_user_action(self, user_id: str, action: str, metadata: dict):
        point = {
            "measurement": "user_actions",
            "tags": {
                "user_id": user_id,
                "action": action
            },
            "fields": metadata,
            "time": datetime.utcnow()
        }
        self.db.write_points([point])
```

### 2. Saga Pattern

Manage data consistency across multiple services using a sequence of local transactions.

```python
from enum import Enum
from abc import ABC, abstractmethod

class SagaStatus(Enum):
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"

class SagaStep(ABC):
    @abstractmethod
    def execute(self, context: dict) -> dict:
        """Execute the step"""
        pass
    
    @abstractmethod
    def compensate(self, context: dict) -> dict:
        """Compensate/rollback the step"""
        pass

class ReserveInventoryStep(SagaStep):
    def __init__(self, inventory_service):
        self.inventory_service = inventory_service
    
    def execute(self, context: dict) -> dict:
        order_items = context['order_items']
        reservation_id = self.inventory_service.reserve_items(order_items)
        return {'reservation_id': reservation_id}
    
    def compensate(self, context: dict) -> dict:
        reservation_id = context.get('reservation_id')
        if reservation_id:
            self.inventory_service.release_reservation(reservation_id)
        return {'compensation': 'inventory_released'}

class ProcessPaymentStep(SagaStep):
    def __init__(self, payment_service):
        self.payment_service = payment_service
    
    def execute(self, context: dict) -> dict:
        payment_details = context['payment_details']
        amount = context['total_amount']
        transaction_id = self.payment_service.charge_payment(payment_details, amount)
        return {'transaction_id': transaction_id}
    
    def compensate(self, context: dict) -> dict:
        transaction_id = context.get('transaction_id')
        if transaction_id:
            self.payment_service.refund_payment(transaction_id)
        return {'compensation': 'payment_refunded'}

class CreateShipmentStep(SagaStep):
    def __init__(self, shipping_service):
        self.shipping_service = shipping_service
    
    def execute(self, context: dict) -> dict:
        order_id = context['order_id']
        shipping_address = context['shipping_address']
        shipment_id = self.shipping_service.create_shipment(order_id, shipping_address)
        return {'shipment_id': shipment_id}
    
    def compensate(self, context: dict) -> dict:
        shipment_id = context.get('shipment_id')
        if shipment_id:
            self.shipping_service.cancel_shipment(shipment_id)
        return {'compensation': 'shipment_cancelled'}

class OrderProcessingSaga:
    """Orchestrates the order processing workflow"""
    
    def __init__(self, inventory_service, payment_service, shipping_service):
        self.steps = [
            ReserveInventoryStep(inventory_service),
            ProcessPaymentStep(payment_service),
            CreateShipmentStep(shipping_service)
        ]
        self.status = SagaStatus.STARTED
        self.context = {}
        self.completed_steps = []
    
    def execute(self, initial_context: dict) -> dict:
        self.context.update(initial_context)
        self.status = SagaStatus.IN_PROGRESS
        
        try:
            for i, step in enumerate(self.steps):
                print(f"Executing step {i + 1}: {step.__class__.__name__}")
                result = step.execute(self.context)
                self.context.update(result)
                self.completed_steps.append(step)
                
            self.status = SagaStatus.COMPLETED
            return {'status': 'success', 'context': self.context}
            
        except Exception as e:
            print(f"Saga failed at step {len(self.completed_steps) + 1}: {e}")
            self.status = SagaStatus.FAILED
            return self._compensate()
    
    def _compensate(self) -> dict:
        """Execute compensation in reverse order"""
        self.status = SagaStatus.COMPENSATING
        compensations = []
        
        # Compensate in reverse order
        for step in reversed(self.completed_steps):
            try:
                print(f"Compensating: {step.__class__.__name__}")
                compensation_result = step.compensate(self.context)
                compensations.append(compensation_result)
            except Exception as e:
                print(f"Compensation failed for {step.__class__.__name__}: {e}")
        
        self.status = SagaStatus.COMPENSATED
        return {
            'status': 'compensated',
            'compensations': compensations,
            'context': self.context
        }

# Usage Example
def process_order(order_data: dict):
    saga = OrderProcessingSaga(
        inventory_service=InventoryService(),
        payment_service=PaymentService(),
        shipping_service=ShippingService()
    )
    
    context = {
        'order_id': order_data['order_id'],
        'order_items': order_data['items'],
        'payment_details': order_data['payment'],
        'total_amount': order_data['total'],
        'shipping_address': order_data['shipping_address']
    }
    
    result = saga.execute(context)
    return result
```

### 3. Circuit Breaker Pattern

Prevent cascading failures by stopping calls to failing services.

```python
import time
from enum import Enum
from collections import deque
from threading import Lock

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreakerError(Exception):
    pass

class CircuitBreaker:
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        self.lock = Lock()
        
        # Track recent calls for monitoring
        self.call_history = deque(maxlen=100)
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    self._record_call(False, "circuit_open")
                    raise CircuitBreakerError("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        self._record_call(True, "success")
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            # After successful call in half-open, close the circuit
            self.state = CircuitState.CLOSED
            self.failure_count = 0
        
        # Reset failure count on success in closed state
        if self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self._record_call(False, "failure")
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def _record_call(self, success: bool, reason: str):
        """Record call for monitoring purposes"""
        self.call_history.append({
            'timestamp': time.time(),
            'success': success,
            'reason': reason,
            'state': self.state.value
        })
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics"""
        recent_calls = list(self.call_history)
        total_calls = len(recent_calls)
        successful_calls = sum(1 for call in recent_calls if call['success'])
        
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'total_calls': total_calls,
            'success_rate': successful_calls / total_calls if total_calls > 0 else 0,
            'last_failure_time': self.last_failure_time
        }

# Service Client with Circuit Breaker
class ResilientServiceClient:
    def __init__(self, service_url: str):
        self.service_url = service_url
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30,
            expected_exception=requests.exceptions.RequestException
        )
    
    def get_user(self, user_id: str) -> Optional[dict]:
        """Get user with circuit breaker protection"""
        try:
            return self.circuit_breaker.call(self._fetch_user, user_id)
        except CircuitBreakerError:
            # Return cached data or default response when circuit is open
            return self._get_cached_user(user_id)
        except requests.exceptions.RequestException:
            # Service call failed
            return None
    
    def _fetch_user(self, user_id: str) -> dict:
        """Actual service call (can fail)"""
        response = requests.get(
            f"{self.service_url}/users/{user_id}",
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    
    def _get_cached_user(self, user_id: str) -> Optional[dict]:
        """Fallback to cached data when circuit is open"""
        # Implementation would fetch from cache
        return {
            'user_id': user_id,
            'name': 'Cached User',
            'source': 'cache'
        }

# Usage Example
user_client = ResilientServiceClient('http://user-service:8080')

# This will work normally when service is healthy
user = user_client.get_user('user123')

# When service starts failing, circuit breaker will eventually open
# and return cached data instead of making failing calls
```

### 4. API Gateway Pattern

Single entry point for all client requests with cross-cutting concerns.

```python
from flask import Flask, request, jsonify, Response
import requests
import json
import jwt
import time
from functools import wraps

class APIGateway:
    def __init__(self):
        self.app = Flask(__name__)
        self.services = {
            'user': 'http://user-service:8080',
            'order': 'http://order-service:8080',
            'product': 'http://product-service:8080',
            'payment': 'http://payment-service:8080'
        }
        self.rate_limits = {}
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API Gateway routes"""
        
        # User service routes
        @self.app.route('/api/users/<user_id>', methods=['GET'])
        @self.require_auth
        @self.rate_limit(requests_per_minute=100)
        def get_user(user_id):
            return self.proxy_request('user', f'/users/{user_id}')
        
        @self.app.route('/api/users', methods=['POST'])
        @self.rate_limit(requests_per_minute=20)
        def create_user():
            return self.proxy_request('user', '/users', method='POST')
        
        # Order service routes
        @self.app.route('/api/orders', methods=['GET'])
        @self.require_auth
        @self.rate_limit(requests_per_minute=50)
        def get_orders():
            user_id = self.get_user_from_token()
            return self.proxy_request('order', f'/orders?user_id={user_id}')
        
        @self.app.route('/api/orders', methods=['POST'])
        @self.require_auth
        @self.rate_limit(requests_per_minute=10)
        def create_order():
            return self.proxy_request('order', '/orders', method='POST')
        
        # Product service routes
        @self.app.route('/api/products', methods=['GET'])
        @self.rate_limit(requests_per_minute=200)
        def get_products():
            return self.proxy_request('product', '/products')
        
        @self.app.route('/api/products/<product_id>', methods=['GET'])
        @self.rate_limit(requests_per_minute=500)
        def get_product(product_id):
            return self.proxy_request('product', f'/products/{product_id}')
    
    def require_auth(self, f):
        """Authentication decorator"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = request.headers.get('Authorization')
            if not token:
                return jsonify({'error': 'No token provided'}), 401
            
            try:
                # Remove 'Bearer ' prefix
                token = token.replace('Bearer ', '')
                payload = jwt.decode(token, 'your-secret-key', algorithms=['HS256'])
                request.user_id = payload['user_id']
                return f(*args, **kwargs)
            except jwt.ExpiredSignatureError:
                return jsonify({'error': 'Token expired'}), 401
            except jwt.InvalidTokenError:
                return jsonify({'error': 'Invalid token'}), 401
        
        return decorated_function
    
    def rate_limit(self, requests_per_minute: int):
        """Rate limiting decorator"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                client_ip = request.remote_addr
                current_time = time.time()
                minute_window = int(current_time // 60)
                
                key = f"{client_ip}:{minute_window}"
                
                if key not in self.rate_limits:
                    self.rate_limits[key] = 0
                
                self.rate_limits[key] += 1
                
                if self.rate_limits[key] > requests_per_minute:
                    return jsonify({
                        'error': 'Rate limit exceeded',
                        'retry_after': 60 - (current_time % 60)
                    }), 429
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def proxy_request(self, service_name: str, path: str, method: str = 'GET') -> Response:
        """Proxy request to appropriate microservice"""
        service_url = self.services.get(service_name)
        if not service_url:
            return jsonify({'error': 'Service not found'}), 404
        
        # Build full URL
        url = f"{service_url}{path}"
        
        # Prepare headers (forward relevant headers)
        headers = {
            'Content-Type': request.content_type or 'application/json',
            'X-Forwarded-For': request.remote_addr,
            'X-Forwarded-Host': request.host,
        }
        
        # Add user context if available
        if hasattr(request, 'user_id'):
            headers['X-User-ID'] = request.user_id
        
        try:
            # Make request to microservice
            if method == 'GET':
                response = requests.get(
                    url,
                    params=request.args,
                    headers=headers,
                    timeout=30
                )
            elif method == 'POST':
                response = requests.post(
                    url,
                    json=request.get_json(),
                    headers=headers,
                    timeout=30
                )
            else:
                return jsonify({'error': 'Method not supported'}), 405
            
            # Return response from microservice
            return Response(
                response.content,
                status=response.status_code,
                headers=dict(response.headers)
            )
            
        except requests.exceptions.RequestException as e:
            return jsonify({
                'error': 'Service unavailable',
                'message': str(e)
            }), 503
    
    def get_user_from_token(self) -> str:
        """Extract user ID from JWT token"""
        return getattr(request, 'user_id', None)
    
    def run(self, host='0.0.0.0', port=8000):
        self.app.run(host=host, port=port)

# Enhanced Gateway with Load Balancing
class LoadBalancedAPIGateway(APIGateway):
    def __init__(self):
        super().__init__()
        # Multiple instances of each service
        self.services = {
            'user': ['http://user-service-1:8080', 'http://user-service-2:8080'],
            'order': ['http://order-service-1:8080', 'http://order-service-2:8080'],
            'product': ['http://product-service-1:8080', 'http://product-service-2:8080']
        }
        self.current_instance = {service: 0 for service in self.services}
    
    def get_service_url(self, service_name: str) -> str:
        """Get service URL with round-robin load balancing"""
        instances = self.services.get(service_name, [])
        if not instances:
            return None
        
        # Round-robin selection
        index = self.current_instance[service_name]
        url = instances[index]
        
        # Move to next instance
        self.current_instance[service_name] = (index + 1) % len(instances)
        
        return url

# Usage
if __name__ == '__main__':
    gateway = LoadBalancedAPIGateway()
    gateway.run()
```

### 5. Event Sourcing Pattern

Store all changes to application state as a sequence of events.

```python
from abc import ABC, abstractmethod
from typing import List, Any, Optional
import json
from datetime import datetime

class DomainEvent(ABC):
    """Base class for all domain events"""
    
    def __init__(self, aggregate_id: str, event_version: int):
        self.aggregate_id = aggregate_id
        self.event_version = event_version
        self.timestamp = datetime.utcnow()
        self.event_id = str(uuid.uuid4())
    
    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize event to dictionary"""
        pass

class AccountCreatedEvent(DomainEvent):
    def __init__(self, aggregate_id: str, event_version: int, owner_name: str, initial_balance: float):
        super().__init__(aggregate_id, event_version)
        self.owner_name = owner_name
        self.initial_balance = initial_balance
    
    def to_dict(self) -> dict:
        return {
            'event_type': 'AccountCreated',
            'aggregate_id': self.aggregate_id,
            'event_version': self.event_version,
            'timestamp': self.timestamp.isoformat(),
            'event_id': self.event_id,
            'data': {
                'owner_name': self.owner_name,
                'initial_balance': self.initial_balance
            }
        }

class MoneyDepositedEvent(DomainEvent):
    def __init__(self, aggregate_id: str, event_version: int, amount: float, deposit_id: str):
        super().__init__(aggregate_id, event_version)
        self.amount = amount
        self.deposit_id = deposit_id
    
    def to_dict(self) -> dict:
        return {
            'event_type': 'MoneyDeposited',
            'aggregate_id': self.aggregate_id,
            'event_version': self.event_version,
            'timestamp': self.timestamp.isoformat(),
            'event_id': self.event_id,
            'data': {
                'amount': self.amount,
                'deposit_id': self.deposit_id
            }
        }

class MoneyWithdrawnEvent(DomainEvent):
    def __init__(self, aggregate_id: str, event_version: int, amount: float, withdrawal_id: str):
        super().__init__(aggregate_id, event_version)
        self.amount = amount
        self.withdrawal_id = withdrawal_id
    
    def to_dict(self) -> dict:
        return {
            'event_type': 'MoneyWithdrawn',
            'aggregate_id': self.aggregate_id,
            'event_version': self.event_version,
            'timestamp': self.timestamp.isoformat(),
            'event_id': self.event_id,
            'data': {
                'amount': self.amount,
                'withdrawal_id': self.withdrawal_id
            }
        }

class EventStore:
    """Event store for persisting and retrieving events"""
    
    def __init__(self):
        self.events = {}  # In production, use a database
    
    def append_events(self, aggregate_id: str, events: List[DomainEvent], expected_version: int):
        """Append events to the store with optimistic concurrency control"""
        if aggregate_id not in self.events:
            self.events[aggregate_id] = []
        
        current_version = len(self.events[aggregate_id])
        if current_version != expected_version:
            raise ConcurrencyError(f"Expected version {expected_version}, but current version is {current_version}")
        
        # Append events
        for event in events:
            self.events[aggregate_id].append(event.to_dict())
    
    def get_events(self, aggregate_id: str, from_version: int = 0) -> List[dict]:
        """Get events for an aggregate from a specific version"""
        if aggregate_id not in self.events:
            return []
        
        return self.events[aggregate_id][from_version:]

class ConcurrencyError(Exception):
    pass

class BankAccount:
    """Event-sourced bank account aggregate"""
    
    def __init__(self, account_id: str):
        self.account_id = account_id
        self.version = 0
        self.owner_name = ""
        self.balance = 0.0
        self.uncommitted_events: List[DomainEvent] = []
    
    @classmethod
    def create(cls, account_id: str, owner_name: str, initial_balance: float = 0.0):
        """Create new bank account"""
        account = cls(account_id)
        
        # Apply event
        event = AccountCreatedEvent(account_id, account.version + 1, owner_name, initial_balance)
        account._apply_event(event)
        account.uncommitted_events.append(event)
        
        return account
    
    def deposit(self, amount: float, deposit_id: str):
        """Deposit money to account"""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        
        event = MoneyDepositedEvent(self.account_id, self.version + 1, amount, deposit_id)
        self._apply_event(event)
        self.uncommitted_events.append(event)
    
    def withdraw(self, amount: float, withdrawal_id: str):
        """Withdraw money from account"""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        
        if self.balance < amount:
            raise ValueError("Insufficient funds")
        
        event = MoneyWithdrawnEvent(self.account_id, self.version + 1, amount, withdrawal_id)
        self._apply_event(event)
        self.uncommitted_events.append(event)
    
    def get_uncommitted_events(self) -> List[DomainEvent]:
        """Get events that haven't been persisted yet"""
        return self.uncommitted_events.copy()
    
    def mark_events_as_committed(self):
        """Mark events as committed (called after successful persistence)"""
        self.uncommitted_events.clear()
    
    def load_from_history(self, events: List[dict]):
        """Rebuild aggregate state from event history"""
        for event_dict in events:
            event = self._deserialize_event(event_dict)
            self._apply_event(event)
    
    def _apply_event(self, event: DomainEvent):
        """Apply event to update aggregate state"""
        if isinstance(event, AccountCreatedEvent):
            self.owner_name = event.owner_name
            self.balance = event.initial_balance
        elif isinstance(event, MoneyDepositedEvent):
            self.balance += event.amount
        elif isinstance(event, MoneyWithdrawnEvent):
            self.balance -= event.amount
        
        self.version = event.event_version
    
    def _deserialize_event(self, event_dict: dict) -> DomainEvent:
        """Convert event dictionary back to event object"""
        event_type = event_dict['event_type']
        data = event_dict['data']
        
        if event_type == 'AccountCreated':
            return AccountCreatedEvent(
                event_dict['aggregate_id'],
                event_dict['event_version'],
                data['owner_name'],
                data['initial_balance']
            )
        elif event_type == 'MoneyDeposited':
            return MoneyDepositedEvent(
                event_dict['aggregate_id'],
                event_dict['event_version'],
                data['amount'],
                data['deposit_id']
            )
        elif event_type == 'MoneyWithdrawn':
            return MoneyWithdrawnEvent(
                event_dict['aggregate_id'],
                event_dict['event_version'],
                data['amount'],
                data['withdrawal_id']
            )
        else:
            raise ValueError(f"Unknown event type: {event_type}")

class BankAccountRepository:
    """Repository for loading and saving bank accounts"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
    
    def get_by_id(self, account_id: str) -> Optional[BankAccount]:
        """Load bank account from event store"""
        events = self.event_store.get_events(account_id)
        if not events:
            return None
        
        account = BankAccount(account_id)
        account.load_from_history(events)
        return account
    
    def save(self, account: BankAccount):
        """Save bank account events to event store"""
        uncommitted_events = account.get_uncommitted_events()
        if not uncommitted_events:
            return
        
        # Calculate expected version (current version minus uncommitted events)
        expected_version = account.version - len(uncommitted_events)
        
        try:
            self.event_store.append_events(
                account.account_id,
                uncommitted_events,
                expected_version
            )
            account.mark_events_as_committed()
        except ConcurrencyError:
            # Handle concurrency conflict
            raise ConcurrencyError("Another process has modified this account")

# Projection for Read Models
class AccountBalanceProjection:
    """Read model projection for account balances"""
    
    def __init__(self):
        self.balances = {}  # In production, use a read database
    
    def handle_account_created(self, event: dict):
        """Handle AccountCreated event"""
        account_id = event['aggregate_id']
        data = event['data']
        
        self.balances[account_id] = {
            'account_id': account_id,
            'owner_name': data['owner_name'],
            'balance': data['initial_balance'],
            'last_updated': event['timestamp']
        }
    
    def handle_money_deposited(self, event: dict):
        """Handle MoneyDeposited event"""
        account_id = event['aggregate_id']
        amount = event['data']['amount']
        
        if account_id in self.balances:
            self.balances[account_id]['balance'] += amount
            self.balances[account_id]['last_updated'] = event['timestamp']
    
    def handle_money_withdrawn(self, event: dict):
        """Handle MoneyWithdrawn event"""
        account_id = event['aggregate_id']
        amount = event['data']['amount']
        
        if account_id in self.balances:
            self.balances[account_id]['balance'] -= amount
            self.balances[account_id]['last_updated'] = event['timestamp']
    
    def get_balance(self, account_id: str) -> Optional[dict]:
        """Get current balance for an account"""
        return self.balances.get(account_id)
    
    def get_all_balances(self) -> List[dict]:
        """Get all account balances"""
        return list(self.balances.values())

# Usage Example
def banking_example():
    # Setup
    event_store = EventStore()
    repository = BankAccountRepository(event_store)
    projection = AccountBalanceProjection()
    
    # Create account
    account = BankAccount.create("acc-123", "John Doe", 100.0)
    repository.save(account)
    
    # Process events in projection
    events = event_store.get_events("acc-123")
    for event in events:
        if event['event_type'] == 'AccountCreated':
            projection.handle_account_created(event)
    
    # Perform transactions
    account = repository.get_by_id("acc-123")
    account.deposit(50.0, "dep-001")
    account.withdraw(25.0, "with-001")
    repository.save(account)
    
    # Update projection
    new_events = event_store.get_events("acc-123", from_version=1)
    for event in new_events:
        if event['event_type'] == 'MoneyDeposited':
            projection.handle_money_deposited(event)
        elif event['event_type'] == 'MoneyWithdrawn':
            projection.handle_money_withdrawn(event)
    
    # Query current balance
    balance_info = projection.get_balance("acc-123")
    print(f"Account balance: {balance_info}")
    # Output: {'account_id': 'acc-123', 'owner_name': 'John Doe', 'balance': 125.0, 'last_updated': '...'}
```

### 6. CQRS (Command Query Responsibility Segregation) Pattern

Separate read and write operations with different models optimized for each purpose.

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

# Commands (Write Side)
class Command(ABC):
    pass

class CreateProductCommand(Command):
    def __init__(self, product_id: str, name: str, price: float, category: str):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.category = category

class UpdateProductPriceCommand(Command):
    def __init__(self, product_id: str, new_price: float):
        self.product_id = product_id
        self.new_price = new_price

class DiscontinueProductCommand(Command):
    def __init__(self, product_id: str, reason: str):
        self.product_id = product_id
        self.reason = reason

# Queries (Read Side)
class Query(ABC):
    pass

class GetProductByIdQuery(Query):
    def __init__(self, product_id: str):
        self.product_id = product_id

class SearchProductsQuery(Query):
    def __init__(self, search_term: str, category: str = None, min_price: float = None, max_price: float = None):
        self.search_term = search_term
        self.category = category
        self.min_price = min_price
        self.max_price = max_price

class GetPopularProductsQuery(Query):
    def __init__(self, limit: int = 10):
        self.limit = limit

# Write Side - Domain Model
class Product:
    def __init__(self, product_id: str, name: str, price: float, category: str):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.category = category
        self.is_active = True
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def update_price(self, new_price: float):
        if new_price <= 0:
            raise ValueError("Price must be positive")
        self.price = new_price
        self.updated_at = datetime.utcnow()
    
    def discontinue(self, reason: str):
        self.is_active = False
        self.updated_at = datetime.utcnow()

# Write Side - Command Handlers
class CommandHandler(ABC):
    @abstractmethod
    def handle(self, command: Command) -> Any:
        pass

class CreateProductCommandHandler(CommandHandler):
    def __init__(self, product_write_repository):
        self.product_repository = product_write_repository
    
    def handle(self, command: CreateProductCommand) -> str:
        # Business logic validation
        if self.product_repository.exists(command.product_id):
            raise ValueError("Product already exists")
        
        # Create domain object
        product = Product(
            command.product_id,
            command.name,
            command.price,
            command.category
        )
        
        # Save to write store
        self.product_repository.save(product)
        
        # Publish event for read side updates
        event = {
            'event_type': 'ProductCreated',
            'product_id': command.product_id,
            'name': command.name,
            'price': command.price,
            'category': command.category,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self._publish_event(event)
        return command.product_id
    
    def _publish_event(self, event: dict):
        # Publish to message bus for eventual consistency
        pass

class UpdateProductPriceCommandHandler(CommandHandler):
    def __init__(self, product_write_repository):
        self.product_repository = product_write_repository
    
    def handle(self, command: UpdateProductPriceCommand) -> None:
        product = self.product_repository.get_by_id(command.product_id)
        if not product:
            raise ValueError("Product not found")
        
        old_price = product.price
        product.update_price(command.new_price)
        self.product_repository.save(product)
        
        # Publish price change event
        event = {
            'event_type': 'ProductPriceUpdated',
            'product_id': command.product_id,
            'old_price': old_price,
            'new_price': command.new_price,
            'timestamp': datetime.utcnow().isoformat()
        }
        self._publish_event(event)

# Read Side - Query Models (Optimized for queries)
class ProductSummaryViewModel:
    def __init__(self, product_id: str, name: str, price: float, category: str):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.category = category

class ProductDetailViewModel:
    def __init__(self, product_id: str, name: str, price: float, category: str, 
                 description: str, rating: float, review_count: int):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.category = category
        self.description = description
        self.rating = rating
        self.review_count = review_count

class PopularProductViewModel:
    def __init__(self, product_id: str, name: str, price: float, sales_count: int):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.sales_count = sales_count

# Read Side - Query Handlers
class QueryHandler(ABC):
    @abstractmethod
    def handle(self, query: Query) -> Any:
        pass

class GetProductByIdQueryHandler(QueryHandler):
    def __init__(self, product_read_repository):
        self.product_read_repository = product_read_repository
    
    def handle(self, query: GetProductByIdQuery) -> Optional[ProductDetailViewModel]:
        # Query optimized read store
        product_data = self.product_read_repository.get_product_details(query.product_id)
        
        if not product_data:
            return None
        
        return ProductDetailViewModel(
            product_id=product_data['product_id'],
            name=product_data['name'],
            price=product_data['price'],
            category=product_data['category'],
            description=product_data['description'],
            rating=product_data['rating'],
            review_count=product_data['review_count']
        )

class SearchProductsQueryHandler(QueryHandler):
    def __init__(self, product_search_repository):
        self.product_search_repository = product_search_repository
    
    def handle(self, query: SearchProductsQuery) -> List[ProductSummaryViewModel]:
        # Use search-optimized store (e.g., Elasticsearch)
        search_results = self.product_search_repository.search(
            term=query.search_term,
            category=query.category,
            price_range=(query.min_price, query.max_price)
        )
        
        return [
            ProductSummaryViewModel(
                product_id=result['id'],
                name=result['name'],
                price=result['price'],
                category=result['category']
            )
            for result in search_results
        ]

class GetPopularProductsQueryHandler(QueryHandler):
    def __init__(self, analytics_repository):
        self.analytics_repository = analytics_repository
    
    def handle(self, query: GetPopularProductsQuery) -> List[PopularProductViewModel]:
        # Query analytics/reporting store
        popular_products = self.analytics_repository.get_top_selling_products(query.limit)
        
        return [
            PopularProductViewModel(
                product_id=product['id'],
                name=product['name'],
                price=product['price'],
                sales_count=product['sales_count']
            )
            for product in popular_products
        ]

# CQRS Bus - Coordinates commands and queries
class CQRSBus:
    def __init__(self):
        self.command_handlers: Dict[type, CommandHandler] = {}
        self.query_handlers: Dict[type, QueryHandler] = {}
    
    def register_command_handler(self, command_type: type, handler: CommandHandler):
        self.command_handlers[command_type] = handler
    
    def register_query_handler(self, query_type: type, handler: QueryHandler):
        self.query_handlers[query_type] = handler
    
    def send_command(self, command: Command) -> Any:
        handler = self.command_handlers.get(type(command))
        if not handler:
            raise ValueError(f"No handler registered for command {type(command).__name__}")
        return handler.handle(command)
    
    def send_query(self, query: Query) -> Any:
        handler = self.query_handlers.get(type(query))
        if not handler:
            raise ValueError(f"No handler registered for query {type(query).__name__}")
        return handler.handle(query)

# Read Model Projections (Event Handlers)
class ProductProjectionHandler:
    """Updates read models when events occur"""
    
    def __init__(self, read_repositories):
        self.product_read_repo = read_repositories['product']
        self.search_repo = read_repositories['search']
        self.analytics_repo = read_repositories['analytics']
    
    def handle_product_created(self, event: dict):
        """Update read models when product is created"""
        # Update detailed view
        self.product_read_repo.create_product_detail({
            'product_id': event['product_id'],
            'name': event['name'],
            'price': event['price'],
            'category': event['category'],
            'description': '',  # Will be updated later
            'rating': 0.0,
            'review_count': 0
        })
        
        # Update search index
        self.search_repo.index_product({
            'id': event['product_id'],
            'name': event['name'],
            'price': event['price'],
            'category': event['category']
        })
        
        # Initialize analytics data
        self.analytics_repo.init_product_stats(event['product_id'])
    
    def handle_product_price_updated(self, event: dict):
        """Update read models when price changes"""
        product_id = event['product_id']
        new_price = event['new_price']
        
        # Update all read models with new price
        self.product_read_repo.update_price(product_id, new_price)
        self.search_repo.update_price(product_id, new_price)
        self.analytics_repo.record_price_change(product_id, new_price)
```

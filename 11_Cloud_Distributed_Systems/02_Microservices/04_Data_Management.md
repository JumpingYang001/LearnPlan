# Data Management in Microservices

*Duration: 2 weeks*

## Overview

Data management in microservices presents unique challenges compared to monolithic applications. This module covers essential patterns and techniques for managing data across distributed services while maintaining consistency, performance, and reliability.

## Key Concepts to Master

### 1. Database per Service Pattern

The **Database per Service** pattern is a fundamental principle in microservices architecture where each service owns and manages its own database. This ensures loose coupling and service autonomy.

#### Core Principles
- **Data Ownership**: Each service is the single source of truth for its data
- **Technology Diversity**: Services can use different database technologies
- **Independent Evolution**: Database schemas can evolve independently
- **Failure Isolation**: Database issues don't cascade across services

#### Implementation Example

**User Service with PostgreSQL:**
```python
# user_service/database.py
import psycopg2
from psycopg2.extras import RealDictCursor
import os

class UserDatabase:
    def __init__(self):
        self.connection = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('USER_DB_NAME', 'users'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'password'),
            cursor_factory=RealDictCursor
        )
    
    def create_user(self, user_data):
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO users (id, email, name, created_at)
                VALUES (%(id)s, %(email)s, %(name)s, %(created_at)s)
                RETURNING *
            """, user_data)
            return cursor.fetchone()
    
    def get_user(self, user_id):
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            return cursor.fetchone()
    
    def update_user(self, user_id, updates):
        set_clause = ", ".join([f"{key} = %({key})s" for key in updates.keys()])
        query = f"UPDATE users SET {set_clause} WHERE id = %(id)s RETURNING *"
        updates['id'] = user_id
        
        with self.connection.cursor() as cursor:
            cursor.execute(query, updates)
            return cursor.fetchone()

# user_service/service.py
from flask import Flask, request, jsonify
from database import UserDatabase
import uuid
from datetime import datetime

app = Flask(__name__)
db = UserDatabase()

@app.route('/users', methods=['POST'])
def create_user():
    user_data = request.json
    user_data['id'] = str(uuid.uuid4())
    user_data['created_at'] = datetime.utcnow()
    
    user = db.create_user(user_data)
    return jsonify(user), 201

@app.route('/users/<user_id>')
def get_user(user_id):
    user = db.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(user)
```

**Product Service with MongoDB:**
```python
# product_service/database.py
from pymongo import MongoClient
from bson import ObjectId
import os

class ProductDatabase:
    def __init__(self):
        self.client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017/'))
        self.db = self.client[os.getenv('PRODUCT_DB_NAME', 'products')]
        self.collection = self.db.products
    
    def create_product(self, product_data):
        result = self.collection.insert_one(product_data)
        product_data['_id'] = str(result.inserted_id)
        return product_data
    
    def get_product(self, product_id):
        product = self.collection.find_one({'_id': ObjectId(product_id)})
        if product:
            product['_id'] = str(product['_id'])
        return product
    
    def search_products(self, query):
        products = list(self.collection.find({
            '$text': {'$search': query}
        }).limit(20))
        
        for product in products:
            product['_id'] = str(product['_id'])
        
        return products
    
    def update_inventory(self, product_id, quantity_change):
        result = self.collection.update_one(
            {'_id': ObjectId(product_id)},
            {'$inc': {'inventory_count': quantity_change}}
        )
        return result.modified_count > 0

# product_service/service.py
from flask import Flask, request, jsonify
from database import ProductDatabase
from datetime import datetime

app = Flask(__name__)
db = ProductDatabase()

@app.route('/products', methods=['POST'])
def create_product():
    product_data = request.json
    product_data['created_at'] = datetime.utcnow()
    
    product = db.create_product(product_data)
    return jsonify(product), 201

@app.route('/products/<product_id>')
def get_product(product_id):
    product = db.get_product(product_id)
    if not product:
        return jsonify({'error': 'Product not found'}), 404
    return jsonify(product)
```

#### Benefits and Challenges

**Benefits:**
- ✅ Service autonomy and independence
- ✅ Technology diversity (SQL, NoSQL, Graph DBs)
- ✅ Scalability per service needs
- ✅ Fault isolation

**Challenges:**
- ❌ Cross-service queries complexity
- ❌ Distributed transactions
- ❌ Data consistency issues
- ❌ Operational overhead

### 2. Eventual Consistency

In distributed systems, **eventual consistency** means that while data may be temporarily inconsistent across services, it will eventually become consistent given enough time and no new updates.

#### Understanding Consistency Levels

```python
# Example: Order Processing with Eventual Consistency

import asyncio
import json
from datetime import datetime
from typing import Dict, List

class EventBus:
    def __init__(self):
        self.subscribers = {}
    
    def subscribe(self, event_type: str, handler):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    async def publish(self, event_type: str, event_data: Dict):
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                try:
                    await handler(event_data)
                except Exception as e:
                    print(f"Event handler error: {e}")

# Global event bus
event_bus = EventBus()

class OrderService:
    def __init__(self):
        self.orders = {}
    
    async def create_order(self, order_data):
        order_id = order_data['order_id']
        order = {
            'id': order_id,
            'user_id': order_data['user_id'],
            'items': order_data['items'],
            'status': 'PENDING',
            'created_at': datetime.utcnow(),
            'total_amount': order_data['total_amount']
        }
        
        self.orders[order_id] = order
        
        # Publish order created event
        await event_bus.publish('order.created', {
            'order_id': order_id,
            'user_id': order['user_id'],
            'items': order['items'],
            'total_amount': order['total_amount']
        })
        
        return order
    
    async def handle_payment_processed(self, event_data):
        order_id = event_data['order_id']
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'PAID'
            self.orders[order_id]['payment_id'] = event_data['payment_id']
            
            # Publish order paid event
            await event_bus.publish('order.paid', {
                'order_id': order_id,
                'payment_id': event_data['payment_id']
            })
    
    async def handle_inventory_reserved(self, event_data):
        order_id = event_data['order_id']
        if order_id in self.orders:
            self.orders[order_id]['inventory_reserved'] = True

class PaymentService:
    def __init__(self):
        self.payments = {}
    
    async def handle_order_created(self, event_data):
        # Simulate payment processing delay
        await asyncio.sleep(2)
        
        payment_id = f"pay_{event_data['order_id']}"
        payment = {
            'id': payment_id,
            'order_id': event_data['order_id'],
            'amount': event_data['total_amount'],
            'status': 'PROCESSED',
            'processed_at': datetime.utcnow()
        }
        
        self.payments[payment_id] = payment
        
        # Publish payment processed event
        await event_bus.publish('payment.processed', {
            'payment_id': payment_id,
            'order_id': event_data['order_id'],
            'amount': event_data['total_amount']
        })

class InventoryService:
    def __init__(self):
        self.inventory = {
            'product_1': 100,
            'product_2': 50,
            'product_3': 25
        }
        self.reservations = {}
    
    async def handle_order_created(self, event_data):
        order_id = event_data['order_id']
        
        # Check if we can reserve inventory
        can_reserve = True
        for item in event_data['items']:
            product_id = item['product_id']
            quantity = item['quantity']
            
            if self.inventory.get(product_id, 0) < quantity:
                can_reserve = False
                break
        
        if can_reserve:
            # Reserve inventory
            for item in event_data['items']:
                product_id = item['product_id']
                quantity = item['quantity']
                self.inventory[product_id] -= quantity
            
            self.reservations[order_id] = event_data['items']
            
            # Publish inventory reserved event
            await event_bus.publish('inventory.reserved', {
                'order_id': order_id,
                'items': event_data['items']
            })
        else:
            # Publish inventory insufficient event
            await event_bus.publish('inventory.insufficient', {
                'order_id': order_id,
                'items': event_data['items']
            })

# Wire up event handlers
async def setup_services():
    order_service = OrderService()
    payment_service = PaymentService()
    inventory_service = InventoryService()
    
    # Subscribe to events
    event_bus.subscribe('order.created', payment_service.handle_order_created)
    event_bus.subscribe('order.created', inventory_service.handle_order_created)
    event_bus.subscribe('payment.processed', order_service.handle_payment_processed)
    event_bus.subscribe('inventory.reserved', order_service.handle_inventory_reserved)
    
    return order_service, payment_service, inventory_service

# Example usage
async def main():
    order_service, payment_service, inventory_service = await setup_services()
    
    # Create an order
    order_data = {
        'order_id': 'order_123',
        'user_id': 'user_456',
        'items': [
            {'product_id': 'product_1', 'quantity': 2},
            {'product_id': 'product_2', 'quantity': 1}
        ],
        'total_amount': 150.00
    }
    
    order = await order_service.create_order(order_data)
    print(f"Order created: {order}")
    
    # Wait for eventual consistency
    await asyncio.sleep(3)
    
    print(f"Final order state: {order_service.orders['order_123']}")
    print(f"Inventory state: {inventory_service.inventory}")

# Run the example
# asyncio.run(main())
```

#### Consistency Patterns

**1. BASE (Basically Available, Soft state, Eventual consistency)**
```python
class BaseConsistencyExample:
    """
    BASE Properties:
    - Basically Available: System remains available despite failures
    - Soft State: Data may change over time due to eventual consistency
    - Eventual Consistency: System will become consistent given enough time
    """
    
    def __init__(self):
        self.user_profiles = {}
        self.user_preferences = {}
        self.pending_updates = []
    
    async def update_user_profile(self, user_id, profile_data):
        # Immediately update primary store (Basically Available)
        self.user_profiles[user_id] = profile_data
        
        # Queue update for related services (Soft State)
        self.pending_updates.append({
            'type': 'profile_update',
            'user_id': user_id,
            'data': profile_data,
            'timestamp': datetime.utcnow()
        })
        
        # Asynchronously propagate changes (Eventual Consistency)
        asyncio.create_task(self._propagate_updates())
        
        return profile_data
    
    async def _propagate_updates(self):
        while self.pending_updates:
            update = self.pending_updates.pop(0)
            try:
                # Simulate propagation to other services
                await self._update_related_services(update)
            except Exception as e:
                # Retry logic for failed updates
                self.pending_updates.append(update)
                await asyncio.sleep(1)  # Backoff
    
    async def _update_related_services(self, update):
        # Simulate updating recommendation service, analytics, etc.
        await asyncio.sleep(0.5)
        print(f"Propagated update: {update['type']} for user {update['user_id']}")
```

### 3. CQRS (Command Query Responsibility Segregation)

CQRS separates read and write operations, allowing for optimized data models and better scalability.

#### Basic CQRS Implementation

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import uuid
from datetime import datetime

# Command Side (Write Model)
class Command(ABC):
    pass

class CreateProductCommand(Command):
    def __init__(self, name: str, price: float, category: str):
        self.id = str(uuid.uuid4())
        self.name = name
        self.price = price
        self.category = category
        self.created_at = datetime.utcnow()

class UpdateProductPriceCommand(Command):
    def __init__(self, product_id: str, new_price: float):
        self.product_id = product_id
        self.new_price = new_price
        self.updated_at = datetime.utcnow()

# Command Handlers
class CommandHandler(ABC):
    @abstractmethod
    async def handle(self, command: Command):
        pass

class ProductCommandHandler(CommandHandler):
    def __init__(self, event_store):
        self.event_store = event_store
    
    async def handle(self, command: Command):
        if isinstance(command, CreateProductCommand):
            event = ProductCreatedEvent(
                product_id=command.id,
                name=command.name,
                price=command.price,
                category=command.category,
                created_at=command.created_at
            )
            await self.event_store.append(event)
            
        elif isinstance(command, UpdateProductPriceCommand):
            event = ProductPriceUpdatedEvent(
                product_id=command.product_id,
                new_price=command.new_price,
                updated_at=command.updated_at
            )
            await self.event_store.append(event)

# Events
class Event(ABC):
    pass

class ProductCreatedEvent(Event):
    def __init__(self, product_id: str, name: str, price: float, category: str, created_at: datetime):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.category = category
        self.created_at = created_at

class ProductPriceUpdatedEvent(Event):
    def __init__(self, product_id: str, new_price: float, updated_at: datetime):
        self.product_id = product_id
        self.new_price = new_price
        self.updated_at = updated_at

# Query Side (Read Model)
class Query(ABC):
    pass

class GetProductQuery(Query):
    def __init__(self, product_id: str):
        self.product_id = product_id

class GetProductsByCategoryQuery(Query):
    def __init__(self, category: str):
        self.category = category

class SearchProductsQuery(Query):
    def __init__(self, search_term: str, max_results: int = 20):
        self.search_term = search_term
        self.max_results = max_results

# Query Handlers and Read Models
class ProductReadModel:
    def __init__(self):
        self.products = {}
        self.category_index = {}
        self.search_index = {}
    
    def apply_event(self, event: Event):
        if isinstance(event, ProductCreatedEvent):
            product = {
                'id': event.product_id,
                'name': event.name,
                'price': event.price,
                'category': event.category,
                'created_at': event.created_at,
                'updated_at': event.created_at
            }
            self.products[event.product_id] = product
            
            # Update category index
            if event.category not in self.category_index:
                self.category_index[event.category] = []
            self.category_index[event.category].append(event.product_id)
            
            # Update search index (simplified)
            words = event.name.lower().split()
            for word in words:
                if word not in self.search_index:
                    self.search_index[word] = []
                self.search_index[word].append(event.product_id)
        
        elif isinstance(event, ProductPriceUpdatedEvent):
            if event.product_id in self.products:
                self.products[event.product_id]['price'] = event.new_price
                self.products[event.product_id]['updated_at'] = event.updated_at

class ProductQueryHandler:
    def __init__(self, read_model: ProductReadModel):
        self.read_model = read_model
    
    async def handle(self, query: Query):
        if isinstance(query, GetProductQuery):
            return self.read_model.products.get(query.product_id)
        
        elif isinstance(query, GetProductsByCategoryQuery):
            product_ids = self.read_model.category_index.get(query.category, [])
            return [self.read_model.products[pid] for pid in product_ids]
        
        elif isinstance(query, SearchProductsQuery):
            matching_product_ids = set()
            words = query.search_term.lower().split()
            
            for word in words:
                if word in self.read_model.search_index:
                    matching_product_ids.update(self.read_model.search_index[word])
            
            results = [self.read_model.products[pid] for pid in matching_product_ids]
            return results[:query.max_results]

# Event Store
class EventStore:
    def __init__(self):
        self.events = []
        self.subscribers = []
    
    async def append(self, event: Event):
        self.events.append(event)
        
        # Notify subscribers (read model projections)
        for subscriber in self.subscribers:
            await subscriber(event)
    
    def subscribe(self, handler):
        self.subscribers.append(handler)

# CQRS Application Service
class ProductService:
    def __init__(self):
        self.event_store = EventStore()
        self.read_model = ProductReadModel()
        self.command_handler = ProductCommandHandler(self.event_store)
        self.query_handler = ProductQueryHandler(self.read_model)
        
        # Subscribe read model to events
        self.event_store.subscribe(self.read_model.apply_event)
    
    async def execute_command(self, command: Command):
        await self.command_handler.handle(command)
    
    async def execute_query(self, query: Query):
        return await self.query_handler.handle(query)

# Usage Example
async def cqrs_example():
    service = ProductService()
    
    # Execute commands (writes)
    create_cmd = CreateProductCommand("Laptop", 999.99, "Electronics")
    await service.execute_command(create_cmd)
    
    update_cmd = UpdateProductPriceCommand(create_cmd.id, 899.99)
    await service.execute_command(update_cmd)
    
    # Execute queries (reads)
    product = await service.execute_query(GetProductQuery(create_cmd.id))
    print(f"Product: {product}")
    
    electronics = await service.execute_query(GetProductsByCategoryQuery("Electronics"))
    print(f"Electronics: {electronics}")
```

#### CQRS Benefits and Considerations

**Benefits:**
- ✅ Optimized read and write models
- ✅ Independent scaling of read/write sides
- ✅ Complex queries without affecting write performance
- ✅ Event-driven architecture support

**Considerations:**
- ❌ Increased complexity
- ❌ Eventual consistency between read/write models
- ❌ Potential code duplication
- ❌ Learning curve for teams

### 4. Event Sourcing

Event Sourcing stores the state of a business entity as a sequence of state-changing events. Instead of storing current state, we store the events that led to that state.

#### Core Concepts

**Event Store Structure:**
```python
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

class EventMetadata:
    def __init__(self, event_id: str = None, timestamp: datetime = None, 
                 correlation_id: str = None, causation_id: str = None):
        self.event_id = event_id or str(uuid.uuid4())
        self.timestamp = timestamp or datetime.utcnow()
        self.correlation_id = correlation_id
        self.causation_id = causation_id

class DomainEvent:
    def __init__(self, aggregate_id: str, event_type: str, event_data: Dict[str, Any], 
                 version: int, metadata: EventMetadata = None):
        self.aggregate_id = aggregate_id
        self.event_type = event_type
        self.event_data = event_data
        self.version = version
        self.metadata = metadata or EventMetadata()
    
    def to_dict(self):
        return {
            'aggregate_id': self.aggregate_id,
            'event_type': self.event_type,
            'event_data': self.event_data,
            'version': self.version,
            'event_id': self.metadata.event_id,
            'timestamp': self.metadata.timestamp.isoformat(),
            'correlation_id': self.metadata.correlation_id,
            'causation_id': self.metadata.causation_id
        }

class EventStore:
    def __init__(self):
        self.events: List[DomainEvent] = []
        self.snapshots: Dict[str, Dict] = {}
    
    async def append_events(self, aggregate_id: str, events: List[DomainEvent], 
                           expected_version: int):
        # Check for concurrency conflicts
        current_version = self.get_current_version(aggregate_id)
        if current_version != expected_version:
            raise ConcurrencyError(f"Expected version {expected_version}, "
                                 f"but current version is {current_version}")
        
        # Append events
        for event in events:
            self.events.append(event)
        
        return len(events)
    
    def get_events(self, aggregate_id: str, from_version: int = 0) -> List[DomainEvent]:
        return [event for event in self.events 
                if event.aggregate_id == aggregate_id and event.version > from_version]
    
    def get_current_version(self, aggregate_id: str) -> int:
        events = [event for event in self.events if event.aggregate_id == aggregate_id]
        return max([event.version for event in events], default=0)
    
    async def save_snapshot(self, aggregate_id: str, snapshot_data: Dict, version: int):
        self.snapshots[aggregate_id] = {
            'data': snapshot_data,
            'version': version,
            'timestamp': datetime.utcnow()
        }
    
    def get_snapshot(self, aggregate_id: str) -> Optional[Dict]:
        return self.snapshots.get(aggregate_id)

class ConcurrencyError(Exception):
    pass
```

#### Bank Account Example with Event Sourcing

```python
from enum import Enum
from decimal import Decimal

class AccountEventType(Enum):
    ACCOUNT_OPENED = "account_opened"
    MONEY_DEPOSITED = "money_deposited"
    MONEY_WITHDRAWN = "money_withdrawn"
    ACCOUNT_CLOSED = "account_closed"

class BankAccount:
    def __init__(self, account_id: str):
        self.account_id = account_id
        self.balance = Decimal('0.00')
        self.is_closed = False
        self.version = 0
        self.uncommitted_events: List[DomainEvent] = []
    
    @classmethod
    def from_events(cls, events: List[DomainEvent]) -> 'BankAccount':
        """Rebuild account state from events"""
        if not events:
            return None
        
        account = cls(events[0].aggregate_id)
        for event in events:
            account.apply_event(event)
        return account
    
    def apply_event(self, event: DomainEvent):
        """Apply an event to update account state"""
        if event.event_type == AccountEventType.ACCOUNT_OPENED.value:
            self.balance = Decimal(str(event.event_data['initial_balance']))
        
        elif event.event_type == AccountEventType.MONEY_DEPOSITED.value:
            self.balance += Decimal(str(event.event_data['amount']))
        
        elif event.event_type == AccountEventType.MONEY_WITHDRAWN.value:
            self.balance -= Decimal(str(event.event_data['amount']))
        
        elif event.event_type == AccountEventType.ACCOUNT_CLOSED.value:
            self.is_closed = True
        
        self.version = event.version
    
    def open_account(self, initial_balance: Decimal, owner_name: str):
        if self.version > 0:
            raise ValueError("Account already exists")
        
        event = DomainEvent(
            aggregate_id=self.account_id,
            event_type=AccountEventType.ACCOUNT_OPENED.value,
            event_data={
                'initial_balance': str(initial_balance),
                'owner_name': owner_name
            },
            version=self.version + 1
        )
        
        self.apply_event(event)
        self.uncommitted_events.append(event)
    
    def deposit(self, amount: Decimal, description: str = ""):
        if self.is_closed:
            raise ValueError("Cannot deposit to closed account")
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        
        event = DomainEvent(
            aggregate_id=self.account_id,
            event_type=AccountEventType.MONEY_DEPOSITED.value,
            event_data={
                'amount': str(amount),
                'description': description,
                'balance_after': str(self.balance + amount)
            },
            version=self.version + 1
        )
        
        self.apply_event(event)
        self.uncommitted_events.append(event)
    
    def withdraw(self, amount: Decimal, description: str = ""):
        if self.is_closed:
            raise ValueError("Cannot withdraw from closed account")
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if self.balance < amount:
            raise ValueError("Insufficient funds")
        
        event = DomainEvent(
            aggregate_id=self.account_id,
            event_type=AccountEventType.MONEY_WITHDRAWN.value,
            event_data={
                'amount': str(amount),
                'description': description,
                'balance_after': str(self.balance - amount)
            },
            version=self.version + 1
        )
        
        self.apply_event(event)
        self.uncommitted_events.append(event)
    
    def close_account(self):
        if self.is_closed:
            raise ValueError("Account already closed")
        if self.balance != Decimal('0.00'):
            raise ValueError("Cannot close account with non-zero balance")
        
        event = DomainEvent(
            aggregate_id=self.account_id,
            event_type=AccountEventType.ACCOUNT_CLOSED.value,
            event_data={
                'closed_at': datetime.utcnow().isoformat()
            },
            version=self.version + 1
        )
        
        self.apply_event(event)
        self.uncommitted_events.append(event)
    
    def get_uncommitted_events(self) -> List[DomainEvent]:
        return self.uncommitted_events.copy()
    
    def mark_events_as_committed(self):
        self.uncommitted_events.clear()

class BankAccountRepository:
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
    
    async def get_by_id(self, account_id: str) -> Optional[BankAccount]:
        # Try to load from snapshot first
        snapshot = self.event_store.get_snapshot(account_id)
        from_version = 0
        
        if snapshot:
            # Load account from snapshot
            account = BankAccount(account_id)
            account.balance = Decimal(snapshot['data']['balance'])
            account.is_closed = snapshot['data']['is_closed']
            account.version = snapshot['version']
            from_version = snapshot['version']
        else:
            account = None
        
        # Load events since snapshot (or all events if no snapshot)
        events = self.event_store.get_events(account_id, from_version)
        
        if not events and not snapshot:
            return None
        
        if not account:
            account = BankAccount.from_events(events)
        else:
            for event in events:
                account.apply_event(event)
        
        return account
    
    async def save(self, account: BankAccount):
        uncommitted_events = account.get_uncommitted_events()
        if not uncommitted_events:
            return
        
        # Save events
        await self.event_store.append_events(
            account.account_id,
            uncommitted_events,
            account.version - len(uncommitted_events)
        )
        
        # Create snapshot every 10 events (optional optimization)
        if account.version % 10 == 0:
            snapshot_data = {
                'balance': str(account.balance),
                'is_closed': account.is_closed
            }
            await self.event_store.save_snapshot(
                account.account_id, snapshot_data, account.version
            )
        
        account.mark_events_as_committed()

# Application Service
class BankAccountService:
    def __init__(self, repository: BankAccountRepository):
        self.repository = repository
    
    async def open_account(self, account_id: str, initial_balance: Decimal, 
                          owner_name: str):
        account = BankAccount(account_id)
        account.open_account(initial_balance, owner_name)
        await self.repository.save(account)
        return account.account_id
    
    async def deposit(self, account_id: str, amount: Decimal, description: str = ""):
        account = await self.repository.get_by_id(account_id)
        if not account:
            raise ValueError("Account not found")
        
        account.deposit(amount, description)
        await self.repository.save(account)
    
    async def withdraw(self, account_id: str, amount: Decimal, description: str = ""):
        account = await self.repository.get_by_id(account_id)
        if not account:
            raise ValueError("Account not found")
        
        account.withdraw(amount, description)
        await self.repository.save(account)
    
    async def get_balance(self, account_id: str) -> Decimal:
        account = await self.repository.get_by_id(account_id)
        if not account:
            raise ValueError("Account not found")
        return account.balance
    
    async def get_account_history(self, account_id: str) -> List[Dict]:
        events = self.repository.event_store.get_events(account_id)
        return [event.to_dict() for event in events]

# Usage Example
async def event_sourcing_example():
    event_store = EventStore()
    repository = BankAccountRepository(event_store)
    service = BankAccountService(repository)
    
    # Open account
    account_id = str(uuid.uuid4())
    await service.open_account(account_id, Decimal('1000.00'), "John Doe")
    
    # Perform transactions
    await service.deposit(account_id, Decimal('500.00'), "Salary")
    await service.withdraw(account_id, Decimal('200.00'), "Rent")
    await service.deposit(account_id, Decimal('100.00'), "Bonus")
    
    # Check balance
    balance = await service.get_balance(account_id)
    print(f"Current balance: {balance}")
    
    # Get full history
    history = await service.get_account_history(account_id)
    for event in history:
        print(f"Event: {event['event_type']}, Data: {event['event_data']}")

# asyncio.run(event_sourcing_example())
```

#### Event Sourcing Benefits and Challenges

**Benefits:**
- ✅ Complete audit trail
- ✅ Temporal queries (state at any point in time)
- ✅ Event replay for debugging
- ✅ Multiple read models from same events
- ✅ Natural fit for event-driven architectures

**Challenges:**
- ❌ Complexity in event schema evolution
- ❌ Query complexity for current state
- ❌ Storage requirements (all events)
- ❌ Eventual consistency considerations

### 5. Distributed Data Management Patterns

#### Saga Pattern for Distributed Transactions

The Saga pattern manages long-running transactions across multiple services through a series of compensating actions.

```python
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any
import asyncio

class SagaStepResult(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"

class SagaStep(ABC):
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> SagaStepResult:
        pass
    
    @abstractmethod
    async def compensate(self, context: Dict[str, Any]) -> bool:
        pass

class OrderProcessingSaga:
    def __init__(self):
        self.steps: List[SagaStep] = [
            ValidateOrderStep(),
            ReserveInventoryStep(),
            ProcessPaymentStep(),
            UpdateInventoryStep(),
            SendNotificationStep()
        ]
        self.completed_steps: List[int] = []
    
    async def execute(self, order_data: Dict[str, Any]) -> bool:
        context = {'order_data': order_data}
        
        try:
            for i, step in enumerate(self.steps):
                print(f"Executing step {i}: {step.__class__.__name__}")
                result = await step.execute(context)
                
                if result == SagaStepResult.SUCCESS:
                    self.completed_steps.append(i)
                elif result == SagaStepResult.FAILURE:
                    await self._compensate(context)
                    return False
                elif result == SagaStepResult.RETRY:
                    # Implement retry logic
                    await asyncio.sleep(1)
                    result = await step.execute(context)
                    if result != SagaStepResult.SUCCESS:
                        await self._compensate(context)
                        return False
                    self.completed_steps.append(i)
            
            print("Saga completed successfully")
            return True
            
        except Exception as e:
            print(f"Saga failed with exception: {e}")
            await self._compensate(context)
            return False
    
    async def _compensate(self, context: Dict[str, Any]):
        print("Starting compensation...")
        # Compensate in reverse order
        for step_index in reversed(self.completed_steps):
            step = self.steps[step_index]
            print(f"Compensating step {step_index}: {step.__class__.__name__}")
            try:
                await step.compensate(context)
            except Exception as e:
                print(f"Compensation failed for step {step_index}: {e}")

# Concrete saga steps
class ValidateOrderStep(SagaStep):
    async def execute(self, context: Dict[str, Any]) -> SagaStepResult:
        order_data = context['order_data']
        
        # Validate order (check required fields, business rules, etc.)
        if not order_data.get('customer_id'):
            return SagaStepResult.FAILURE
        
        if not order_data.get('items'):
            return SagaStepResult.FAILURE
        
        context['validated'] = True
        return SagaStepResult.SUCCESS
    
    async def compensate(self, context: Dict[str, Any]) -> bool:
        # Nothing to compensate for validation
        return True

class ReserveInventoryStep(SagaStep):
    async def execute(self, context: Dict[str, Any]) -> SagaStepResult:
        order_data = context['order_data']
        
        # Simulate inventory reservation
        await asyncio.sleep(1)
        
        # Check if items are available
        for item in order_data['items']:
            if item['quantity'] > 10:  # Simulate out of stock
                return SagaStepResult.FAILURE
        
        # Reserve inventory
        reservation_id = f"reservation_{order_data['order_id']}"
        context['reservation_id'] = reservation_id
        
        print(f"Reserved inventory: {reservation_id}")
        return SagaStepResult.SUCCESS
    
    async def compensate(self, context: Dict[str, Any]) -> bool:
        if 'reservation_id' in context:
            print(f"Releasing inventory reservation: {context['reservation_id']}")
            # Release inventory reservation
            await asyncio.sleep(0.5)
        return True

class ProcessPaymentStep(SagaStep):
    async def execute(self, context: Dict[str, Any]) -> SagaStepResult:
        order_data = context['order_data']
        
        # Simulate payment processing
        await asyncio.sleep(2)
        
        # Simulate payment failure for high amounts
        total_amount = sum(item['price'] * item['quantity'] for item in order_data['items'])
        if total_amount > 1000:
            return SagaStepResult.FAILURE
        
        payment_id = f"payment_{order_data['order_id']}"
        context['payment_id'] = payment_id
        
        print(f"Payment processed: {payment_id}")
        return SagaStepResult.SUCCESS
    
    async def compensate(self, context: Dict[str, Any]) -> bool:
        if 'payment_id' in context:
            print(f"Refunding payment: {context['payment_id']}")
            # Process refund
            await asyncio.sleep(1)
        return True

class UpdateInventoryStep(SagaStep):
    async def execute(self, context: Dict[str, Any]) -> SagaStepResult:
        order_data = context['order_data']
        
        # Update actual inventory
        await asyncio.sleep(1)
        
        print("Inventory updated")
        context['inventory_updated'] = True
        return SagaStepResult.SUCCESS
    
    async def compensate(self, context: Dict[str, Any]) -> bool:
        if context.get('inventory_updated'):
            print("Reverting inventory changes")
            # Revert inventory changes
            await asyncio.sleep(0.5)
        return True

class SendNotificationStep(SagaStep):
    async def execute(self, context: Dict[str, Any]) -> SagaStepResult:
        order_data = context['order_data']
        
        # Send confirmation notification
        await asyncio.sleep(0.5)
        
        print(f"Notification sent to customer {order_data['customer_id']}")
        context['notification_sent'] = True
        return SagaStepResult.SUCCESS
    
    async def compensate(self, context: Dict[str, Any]) -> bool:
        if context.get('notification_sent'):
            print("Sending cancellation notification")
            # Send cancellation notification
            await asyncio.sleep(0.5)
        return True

# Saga orchestrator service
class OrderSagaOrchestrator:
    async def process_order(self, order_data: Dict[str, Any]) -> bool:
        saga = OrderProcessingSaga()
        return await saga.execute(order_data)

# Usage example
async def saga_example():
    orchestrator = OrderSagaOrchestrator()
    
    # Successful order
    order_data = {
        'order_id': 'order_123',
        'customer_id': 'customer_456',
        'items': [
            {'product_id': 'product_1', 'quantity': 2, 'price': 100},
            {'product_id': 'product_2', 'quantity': 1, 'price': 50}
        ]
    }
    
    print("Processing successful order...")
    success = await orchestrator.process_order(order_data)
    print(f"Order processing result: {success}\n")
    
    # Failed order (high amount triggers payment failure)
    failed_order_data = {
        'order_id': 'order_124',
        'customer_id': 'customer_457',
        'items': [
            {'product_id': 'product_3', 'quantity': 5, 'price': 300}
        ]
    }
    
    print("Processing failed order...")
    success = await orchestrator.process_order(failed_order_data)
    print(f"Order processing result: {success}")

# asyncio.run(saga_example())
```

#### Data Synchronization Patterns

**1. Change Data Capture (CDC)**
```python
import asyncio
from typing import Dict, Any, List
import json

class CDCEvent:
    def __init__(self, table: str, operation: str, before: Dict, after: Dict, timestamp: str):
        self.table = table
        self.operation = operation  # INSERT, UPDATE, DELETE
        self.before = before
        self.after = after
        self.timestamp = timestamp

class CDCProcessor:
    def __init__(self):
        self.subscribers = {}
    
    def subscribe(self, table: str, handler):
        if table not in self.subscribers:
            self.subscribers[table] = []
        self.subscribers[table].append(handler)
    
    async def process_event(self, event: CDCEvent):
        if event.table in self.subscribers:
            for handler in self.subscribers[event.table]:
                try:
                    await handler(event)
                except Exception as e:
                    print(f"CDC handler error: {e}")

# Example CDC handlers
class UserProfileSyncHandler:
    def __init__(self, search_service_client):
        self.search_service_client = search_service_client
    
    async def handle_user_change(self, event: CDCEvent):
        if event.operation == 'INSERT':
            await self.search_service_client.index_user(event.after)
        elif event.operation == 'UPDATE':
            await self.search_service_client.update_user(event.after)
        elif event.operation == 'DELETE':
            await self.search_service_client.delete_user(event.before['id'])

class CacheInvalidationHandler:
    def __init__(self, cache_client):
        self.cache_client = cache_client
    
    async def handle_product_change(self, event: CDCEvent):
        if event.operation in ['UPDATE', 'DELETE']:
            product_id = event.before.get('id') or event.after.get('id')
            cache_key = f"product:{product_id}"
            await self.cache_client.delete(cache_key)

# Mock services for demonstration
class SearchServiceClient:
    async def index_user(self, user_data):
        print(f"Indexing user in search service: {user_data['id']}")
    
    async def update_user(self, user_data):
        print(f"Updating user in search service: {user_data['id']}")
    
    async def delete_user(self, user_id):
        print(f"Deleting user from search service: {user_id}")

class CacheClient:
    async def delete(self, key):
        print(f"Invalidating cache key: {key}")
```

## Learning Objectives

By the end of this section, you should be able to:

### Core Understanding
- **Explain the Database per Service pattern** and its implications for microservices architecture
- **Implement eventual consistency** strategies in distributed systems
- **Design and implement CQRS** (Command Query Responsibility Segregation) architecture
- **Apply Event Sourcing** for audit trails and temporal queries
- **Manage distributed transactions** using Saga pattern and other coordination techniques

### Practical Skills
- **Design data models** that work across service boundaries
- **Implement data synchronization** between services using various patterns
- **Handle concurrency** and consistency challenges in distributed environments
- **Build resilient data pipelines** with proper error handling and retry mechanisms
- **Monitor and debug** data consistency issues in production systems

### Self-Assessment Checklist

Before proceeding to the next section, ensure you can:

□ Design a microservices system with proper database separation  
□ Implement a simple CQRS system with separate read/write models  
□ Build an event sourcing system with event replay capabilities  
□ Create a saga orchestration for distributed transactions  
□ Handle eventual consistency scenarios in practice  
□ Implement outbox pattern for reliable event publishing  
□ Design data synchronization between services  
□ Debug consistency issues in distributed systems  

### Practical Exercises

**Exercise 1: E-commerce Data Design**
```python
# TODO: Design a microservices data architecture for an e-commerce system
# Services: User, Product, Order, Payment, Inventory
# Requirements:
# - Each service has its own database
# - Orders need to validate inventory and process payments
# - Users should see consistent order status
# - Product catalog should be searchable
```

**Exercise 2: Event Sourcing Implementation**
```python
# TODO: Implement an event-sourced shopping cart
# Events: CartCreated, ItemAdded, ItemRemoved, ItemQuantityChanged, CartAbandoned
# Requirements:
# - Rebuild cart state from events
# - Support snapshots for performance
# - Handle concurrent modifications
```

**Exercise 3: Saga Implementation**
```python
# TODO: Implement a hotel booking saga
# Steps: ValidateAvailability, ReserveRoom, ProcessPayment, ConfirmBooking, SendConfirmation
# Requirements:
# - Handle failures with compensation
# - Support retry logic for transient failures
# - Provide booking status tracking
```

**Exercise 4: CQRS with Read Models**
```python
# TODO: Build a blog system with CQRS
# Commands: CreatePost, UpdatePost, DeletePost, AddComment
# Read Models: PostList, PostDetail, CommentsByPost, PopularPosts
# Requirements:
# - Separate command and query sides
# - Multiple optimized read models
# - Event-driven read model updates
```

## Study Materials

### Recommended Reading

**Primary Resources:**
- **"Microservices Patterns" by Chris Richardson** - Chapters 4-5 (Data management patterns)
- **"Building Event-Driven Microservices" by Adam Bellemare** - Event sourcing and streaming
- **"Designing Data-Intensive Applications" by Martin Kleppmann** - Distributed data concepts

**Online Resources:**
- [Microservices.io - Data Management Patterns](https://microservices.io/patterns/data/)
- [Martin Fowler - CQRS](https://martinfowler.com/bliki/CQRS.html)
- [Event Sourcing Pattern - Microsoft Docs](https://docs.microsoft.com/en-us/azure/architecture/patterns/event-sourcing)
- [Saga Pattern - Microsoft Docs](https://docs.microsoft.com/en-us/azure/architecture/reference-architectures/saga/saga)

**Video Courses:**
- "Event Sourcing and CQRS" - Pluralsight
- "Microservices Data Patterns" - Udemy
- "Distributed Systems" - MIT OpenCourseWare

### Hands-on Labs

**Lab 1: Database per Service Setup**
- Set up multiple services with different databases (PostgreSQL, MongoDB, Redis)
- Implement service-specific data access patterns
- Handle cross-service data requirements

**Lab 2: Event Sourcing Workshop**
- Build an event store from scratch
- Implement aggregate reconstruction from events
- Add snapshot support for performance

**Lab 3: CQRS Implementation**
- Create command and query models for a domain
- Implement read model projections
- Handle eventual consistency between models

**Lab 4: Saga Orchestration**
- Build a distributed transaction using saga pattern
- Implement compensation logic
- Add monitoring and observability

### Practice Questions

**Conceptual Questions:**
1. What are the trade-offs between shared databases and database per service?
2. How does eventual consistency differ from strong consistency? When would you choose each?
3. What are the benefits and drawbacks of CQRS? When should you use it?
4. How does event sourcing help with audit trails and compliance requirements?
5. What is the difference between orchestration and choreography in sagas?

**Design Questions:**
6. Design a data architecture for a social media platform with multiple services
7. How would you handle user profile updates that need to be reflected in multiple services?
8. Design an event sourcing system for a financial trading application
9. How would you implement search functionality across multiple microservices?
10. Design a data synchronization strategy for a mobile app with offline capabilities

**Technical Challenges:**
```python
# Challenge 1: Implement optimistic concurrency control
class OptimisticLockingRepository:
    async def save_with_version_check(self, entity, expected_version):
        # TODO: Implement version-based optimistic locking
        pass

# Challenge 2: Build an event replay system
class EventReplayService:
    async def replay_events_from_date(self, aggregate_id, from_date):
        # TODO: Replay events from a specific date
        pass

# Challenge 3: Implement cross-service transaction coordination
class TransactionCoordinator:
    async def coordinate_transaction(self, transaction_steps):
        # TODO: Coordinate distributed transaction with 2PC or saga
        pass
```

### Technology Stack and Tools

**Databases:**
```bash
# Install and setup different database types
docker run -d --name postgres -e POSTGRES_PASSWORD=password -p 5432:5432 postgres:13
docker run -d --name mongodb -p 27017:27017 mongo:4.4
docker run -d --name redis -p 6379:6379 redis:6-alpine
```

**Event Streaming:**
```bash
# Apache Kafka for event streaming
docker run -d --name kafka -p 9092:9092 \
  -e KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 \
  -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
  confluentinc/cp-kafka:latest

# Redis Streams alternative
redis-cli XADD events * event order_created data '{"order_id":"123"}'
```

**Development Tools:**
```python
# Install required Python packages
pip install asyncpg pymongo redis kafka-python sqlalchemy alembic

# Database migration tool
alembic init migrations
alembic revision --autogenerate -m "Initial migration"
alembic upgrade head
```

**Monitoring and Observability:**
```python
# Example monitoring setup
import logging
import time
from functools import wraps

def monitor_data_consistency(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            logging.info(f"Data operation {func.__name__} completed in {time.time() - start_time:.2f}s")
            return result
        except Exception as e:
            logging.error(f"Data operation {func.__name__} failed: {e}")
            raise
    return wrapper

# Usage
@monitor_data_consistency
async def create_order(order_data):
    # Order creation logic
    pass
```

### Best Practices and Patterns

**Data Consistency Patterns:**
1. **Eventual Consistency with Compensation** - For non-critical data
2. **Strong Consistency with Distributed Locks** - For critical business data
3. **Read-your-writes Consistency** - For user experience
4. **Causal Consistency** - For related operations

**Event Design Patterns:**
1. **Event Versioning** - Handle schema evolution
2. **Event Upcasting** - Migrate old events to new formats
3. **Event Compaction** - Optimize storage for snapshot generation
4. **Event Filtering** - Subscribe only to relevant events

**Performance Optimization:**
```python
# Connection pooling
from sqlalchemy.pool import QueuePool

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)

# Read replicas for queries
class DatabaseRouter:
    def __init__(self, write_db, read_replicas):
        self.write_db = write_db
        self.read_replicas = read_replicas
    
    def get_read_connection(self):
        # Load balance across read replicas
        return random.choice(self.read_replicas)
    
    def get_write_connection(self):
        return self.write_db
```

**Error Handling and Resilience:**
```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

class ResilientDataService:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def save_with_retry(self, data):
        try:
            return await self.database.save(data)
        except ConnectionError:
            # Retry on connection errors
            raise
        except ValidationError:
            # Don't retry on validation errors
            raise
    
    async def save_with_circuit_breaker(self, data):
        if self.circuit_breaker.is_open():
            raise CircuitBreakerOpenError("Database circuit breaker is open")
        
        try:
            result = await self.database.save(data)
            self.circuit_breaker.record_success()
            return result
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise
```

## Common Patterns and Anti-patterns

### ✅ Recommended Patterns

1. **Database per Service** - Each service owns its data
2. **Event-driven Updates** - Use events for cross-service data synchronization
3. **CQRS for Complex Domains** - Separate read/write models when beneficial
4. **Saga for Distributed Transactions** - Coordinate long-running transactions
5. **Outbox Pattern** - Ensure reliable event publishing

### ❌ Anti-patterns to Avoid

1. **Shared Database** - Multiple services accessing the same database
2. **Distributed Transactions (2PC)** - Avoid distributed ACID transactions
3. **Synchronous Cross-service Calls for Data** - Creates tight coupling
4. **Inconsistent Data Models** - Same entity with different representations
5. **Event Sourcing Everything** - Only use where beneficial

### Migration Strategies

**From Monolith to Microservices:**
```python
# Strangler Fig Pattern for gradual migration
class DataMigrationService:
    def __init__(self, old_db, new_service_client):
        self.old_db = old_db
        self.new_service_client = new_service_client
    
    async def read_user(self, user_id):
        # Try new service first
        try:
            return await self.new_service_client.get_user(user_id)
        except UserNotFoundError:
            # Fallback to old database
            return await self.old_db.get_user(user_id)
    
    async def write_user(self, user_data):
        # Write to both old and new systems during migration
        await self.old_db.save_user(user_data)
        await self.new_service_client.create_user(user_data)
```

## Next Steps

After mastering these data management patterns, you should explore:
- **Advanced Event Streaming** with Apache Kafka or Pulsar
- **Distributed Query Engines** like Presto or Apache Drill
- **Data Mesh Architecture** for large-scale data management
- **Cloud-native Data Services** (AWS RDS, Azure Cosmos DB, Google Cloud Spanner)
- **Data Governance and Compliance** in distributed systems


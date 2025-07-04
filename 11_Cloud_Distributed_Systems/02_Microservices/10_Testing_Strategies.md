# Testing Strategies for Microservices

*Duration: 2-3 weeks*

## Overview

Testing microservices presents unique challenges compared to monolithic applications. With distributed systems, you need to ensure not only that individual services work correctly but also that they interact properly with each other across network boundaries. This guide covers comprehensive testing strategies from unit tests to end-to-end testing.

## Testing Pyramid for Microservices

The testing pyramid for microservices has evolved to address distributed system complexities:

```
                     E2E Tests
                   ┌─────────────┐
                   │   Slow      │
                   │ Expensive   │ 
                   │  Brittle    │
                   └─────────────┘
                 ┌─────────────────┐
                 │ Contract Tests  │
                 │   Fast(er)      │
                 │  Reliable       │ 
                 └─────────────────┘
               ┌─────────────────────┐
               │ Integration Tests   │
               │     Medium          │
               │   Complexity        │
               └─────────────────────┘
             ┌─────────────────────────┐
             │     Unit Tests          │
             │       Fast              │
             │      Cheap              │
             │     Reliable            │
             └─────────────────────────┘
```

## 1. Unit Testing

Unit tests focus on testing individual components in isolation, mocking external dependencies.

### Basic Unit Test Structure

```python
import unittest
from unittest.mock import Mock, patch
import pytest
from src.user_service import UserService
from src.models import User
from src.exceptions import UserNotFoundError

class TestUserService(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_db = Mock()
        self.mock_email_service = Mock()
        self.user_service = UserService(
            database=self.mock_db,
            email_service=self.mock_email_service
        )
    
    def test_create_user_success(self):
        """Test successful user creation."""
        # Arrange
        user_data = {
            'name': 'John Doe',
            'email': 'john@example.com',
            'age': 30
        }
        expected_user = User(id=1, **user_data)
        self.mock_db.save.return_value = expected_user
        
        # Act
        result = self.user_service.create_user(user_data)
        
        # Assert
        self.assertEqual(result.name, 'John Doe')
        self.assertEqual(result.email, 'john@example.com')
        self.mock_db.save.assert_called_once()
        self.mock_email_service.send_welcome_email.assert_called_once_with('john@example.com')
    
    def test_create_user_with_invalid_email(self):
        """Test user creation with invalid email format."""
        # Arrange
        user_data = {
            'name': 'John Doe',
            'email': 'invalid-email',
            'age': 30
        }
        
        # Act & Assert
        with self.assertRaises(ValueError) as context:
            self.user_service.create_user(user_data)
        
        self.assertIn('Invalid email format', str(context.exception))
        self.mock_db.save.assert_not_called()
    
    def test_get_user_not_found(self):
        """Test getting a user that doesn't exist."""
        # Arrange
        user_id = 999
        self.mock_db.find_by_id.return_value = None
        
        # Act & Assert
        with self.assertRaises(UserNotFoundError):
            self.user_service.get_user(user_id)
    
    @patch('src.user_service.datetime')
    def test_user_age_calculation(self, mock_datetime):
        """Test age calculation with mocked current date."""
        # Arrange
        from datetime import datetime
        mock_datetime.now.return_value = datetime(2025, 1, 1)
        
        user_data = {
            'name': 'Jane Doe',
            'birth_date': '1990-01-01'
        }
        
        # Act
        age = self.user_service.calculate_age(user_data['birth_date'])
        
        # Assert
        self.assertEqual(age, 35)

if __name__ == '__main__':
    unittest.main()
```

### Advanced Unit Testing with pytest

```python
import pytest
from unittest.mock import Mock, AsyncMock
import asyncio
from src.async_user_service import AsyncUserService

@pytest.fixture
def mock_database():
    """Fixture providing a mock database."""
    return Mock()

@pytest.fixture
def mock_cache():
    """Fixture providing a mock cache service."""
    cache = Mock()
    cache.get = AsyncMock()
    cache.set = AsyncMock()
    return cache

@pytest.fixture
def user_service(mock_database, mock_cache):
    """Fixture providing UserService with mocked dependencies."""
    return AsyncUserService(
        database=mock_database,
        cache=mock_cache
    )

class TestAsyncUserService:
    
    @pytest.mark.asyncio
    async def test_get_user_from_cache(self, user_service, mock_cache):
        """Test retrieving user from cache."""
        # Arrange
        user_id = 123
        cached_user = {'id': 123, 'name': 'Cached User'}
        mock_cache.get.return_value = cached_user
        
        # Act
        result = await user_service.get_user(user_id)
        
        # Assert
        assert result == cached_user
        mock_cache.get.assert_called_once_with(f'user:{user_id}')
    
    @pytest.mark.asyncio
    async def test_get_user_cache_miss(self, user_service, mock_database, mock_cache):
        """Test retrieving user when not in cache."""
        # Arrange
        user_id = 123
        db_user = {'id': 123, 'name': 'DB User'}
        mock_cache.get.return_value = None
        mock_database.find_by_id.return_value = db_user
        
        # Act
        result = await user_service.get_user(user_id)
        
        # Assert
        assert result == db_user
        mock_cache.set.assert_called_once_with(f'user:{user_id}', db_user, ttl=3600)
    
    @pytest.mark.parametrize("email,expected", [
        ("valid@example.com", True),
        ("invalid-email", False),
        ("", False),
        ("test@", False),
        ("@example.com", False),
    ])
    def test_email_validation(self, user_service, email, expected):
        """Test email validation with various inputs."""
        result = user_service.validate_email(email)
        assert result == expected
```

## 2. Integration Testing

Integration tests verify that different components work together correctly, often involving real databases, message queues, or external services.

### Database Integration Tests

```python
import pytest
import asyncpg
from testcontainers.postgres import PostgresContainer
from src.database import Database
from src.user_repository import UserRepository

class TestUserRepositoryIntegration:
    
    @pytest.fixture(scope="class")
    def postgres_container(self):
        """Start PostgreSQL container for testing."""
        with PostgresContainer("postgres:13") as postgres:
            yield postgres
    
    @pytest.fixture(scope="class")
    async def database(self, postgres_container):
        """Set up database connection and schema."""
        connection_url = postgres_container.get_connection_url()
        db = Database(connection_url)
        await db.connect()
        
        # Create schema
        await db.execute("""
            CREATE TABLE users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        yield db
        await db.disconnect()
    
    @pytest.fixture
    async def user_repository(self, database):
        """Create UserRepository instance."""
        return UserRepository(database)
    
    @pytest.mark.asyncio
    async def test_create_and_retrieve_user(self, user_repository):
        """Test creating and retrieving a user from database."""
        # Arrange
        user_data = {
            'name': 'Integration Test User',
            'email': 'integration@example.com'
        }
        
        # Act
        created_user = await user_repository.create(user_data)
        retrieved_user = await user_repository.get_by_id(created_user.id)
        
        # Assert
        assert retrieved_user.id == created_user.id
        assert retrieved_user.name == user_data['name']
        assert retrieved_user.email == user_data['email']
    
    @pytest.mark.asyncio
    async def test_duplicate_email_constraint(self, user_repository):
        """Test that duplicate emails are rejected."""
        # Arrange
        user_data = {
            'name': 'Test User',
            'email': 'duplicate@example.com'
        }
        
        # Act
        await user_repository.create(user_data)
        
        # Assert
        with pytest.raises(asyncpg.UniqueViolationError):
            await user_repository.create(user_data)
```

### Message Queue Integration Tests

```python
import pytest
import asyncio
from testcontainers.rabbitmq import RabbitMqContainer
from src.message_queue import MessageQueue
from src.order_service import OrderService
from src.inventory_service import InventoryService

class TestMessageQueueIntegration:
    
    @pytest.fixture(scope="class")
    def rabbitmq_container(self):
        """Start RabbitMQ container for testing."""
        with RabbitMqContainer() as rabbitmq:
            yield rabbitmq
    
    @pytest.fixture
    async def message_queue(self, rabbitmq_container):
        """Set up message queue connection."""
        connection_url = rabbitmq_container.get_connection_url()
        mq = MessageQueue(connection_url)
        await mq.connect()
        yield mq
        await mq.disconnect()
    
    @pytest.mark.asyncio
    async def test_order_inventory_integration(self, message_queue):
        """Test order service communicating with inventory service via queue."""
        # Arrange
        order_service = OrderService(message_queue)
        inventory_service = InventoryService(message_queue)
        
        # Start inventory service listener
        await inventory_service.start_listening()
        
        # Act
        order_result = await order_service.place_order({
            'product_id': 'PROD123',
            'quantity': 5,
            'customer_id': 'CUST456'
        })
        
        # Wait for message processing
        await asyncio.sleep(1)
        
        # Assert
        assert order_result['status'] == 'pending'
        
        # Check that inventory was updated
        inventory = await inventory_service.get_inventory('PROD123')
        assert inventory['reserved'] == 5
```

## 3. Contract Testing

Contract testing ensures that the communication contracts between services are maintained. We'll use Pact for consumer-driven contract testing.

### Consumer Contract Test

```python
import pytest
from pact import Consumer, Provider, Like, Term
from src.user_client import UserClient

# Consumer: Order Service
# Provider: User Service

@pytest.fixture(scope="module")
def pact():
    """Set up Pact consumer contract."""
    return Consumer('order-service').has_pact_with(Provider('user-service'))

class TestUserServiceContract:
    
    def test_get_user_exists(self, pact):
        """Test contract for getting an existing user."""
        # Define expected interaction
        (pact
         .given('user with id 123 exists')
         .upon_receiving('a request for user 123')
         .with_request(
             method='GET',
             path='/users/123',
             headers={'Accept': 'application/json'}
         )
         .will_respond_with(
             status=200,
             headers={'Content-Type': 'application/json'},
             body=Like({
                 'id': 123,
                 'name': 'John Doe',
                 'email': Term(r'.+@.+\..+', 'john@example.com'),
                 'created_at': Term(
                     r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z',
                     '2025-01-01T10:00:00Z'
                 )
             })
         ))
        
        # Test the interaction
        with pact:
            user_client = UserClient(pact.uri)
            user = user_client.get_user(123)
            
            assert user['id'] == 123
            assert user['name'] == 'John Doe'
            assert '@' in user['email']
    
    def test_get_user_not_found(self, pact):
        """Test contract for getting a non-existent user."""
        (pact
         .given('user with id 999 does not exist')
         .upon_receiving('a request for user 999')
         .with_request(
             method='GET',
             path='/users/999',
             headers={'Accept': 'application/json'}
         )
         .will_respond_with(
             status=404,
             headers={'Content-Type': 'application/json'},
             body={
                 'error': 'User not found',
                 'code': 'USER_NOT_FOUND'
             }
         ))
        
        with pact:
            user_client = UserClient(pact.uri)
            
            with pytest.raises(UserNotFoundError) as exc_info:
                user_client.get_user(999)
            
            assert exc_info.value.code == 'USER_NOT_FOUND'
```

### Provider Contract Verification

```python
import pytest
from pact import Verifier
from src.app import create_app
import threading
import time

class TestUserServiceProviderContract:
    
    @pytest.fixture(scope="class")
    def provider_app(self):
        """Start the provider application for contract verification."""
        app = create_app()
        
        # Start the app in a separate thread
        def run_app():
            app.run(host='127.0.0.1', port=5001, debug=False)
        
        thread = threading.Thread(target=run_app, daemon=True)
        thread.start()
        time.sleep(2)  # Wait for app to start
        
        yield "http://127.0.0.1:5001"
    
    def test_user_service_honors_contract(self, provider_app):
        """Verify that the user service honors the contract."""
        verifier = Verifier(
            provider='user-service',
            provider_base_url=provider_app
        )
        
        # Provider states setup
        def provider_state_setup(state):
            if state == 'user with id 123 exists':
                # Set up test data
                from src.test_data import create_test_user
                create_test_user(id=123, name='John Doe')
            elif state == 'user with id 999 does not exist':
                # Ensure user doesn't exist
                from src.test_data import delete_user
                delete_user(999)
        
        # Verify the contract
        success, logs = verifier.verify_with_broker(
            broker_base_url="http://pact-broker:9292",
            broker_username="pact",
            broker_password="pact",
            provider_states_setup_url=f"{provider_app}/pact/provider-states",
            provider_version="1.0.0"
        )
        
        assert success, f"Contract verification failed: {logs}"
```

## 4. End-to-End Testing

E2E tests verify complete user journeys across multiple services.

### API End-to-End Tests

```python
import pytest
import requests
import asyncio
from testcontainers.compose import DockerCompose
import json

class TestE2EUserJourney:
    
    @pytest.fixture(scope="class")
    def services(self):
        """Start all services using docker-compose."""
        with DockerCompose(".", compose_file_name="docker-compose.test.yml") as compose:
            # Wait for services to be ready
            compose.wait_for("http://localhost:8080/health")
            compose.wait_for("http://localhost:8081/health")
            compose.wait_for("http://localhost:8082/health")
            
            yield {
                'user_service': 'http://localhost:8080',
                'order_service': 'http://localhost:8081',
                'inventory_service': 'http://localhost:8082'
            }
    
    def test_complete_order_flow(self, services):
        """Test complete order flow from user creation to order fulfillment."""
        
        # Step 1: Create a new user
        user_data = {
            'name': 'E2E Test User',
            'email': 'e2e@example.com',
            'address': '123 Test St, Test City'
        }
        
        user_response = requests.post(
            f"{services['user_service']}/users",
            json=user_data
        )
        assert user_response.status_code == 201
        user = user_response.json()
        user_id = user['id']
        
        # Step 2: Add product to inventory
        product_data = {
            'id': 'PROD123',
            'name': 'Test Product',
            'price': 29.99,
            'quantity': 100
        }
        
        inventory_response = requests.post(
            f"{services['inventory_service']}/products",
            json=product_data
        )
        assert inventory_response.status_code == 201
        
        # Step 3: Place an order
        order_data = {
            'user_id': user_id,
            'items': [
                {
                    'product_id': 'PROD123',
                    'quantity': 2
                }
            ]
        }
        
        order_response = requests.post(
            f"{services['order_service']}/orders",
            json=order_data
        )
        assert order_response.status_code == 201
        order = order_response.json()
        order_id = order['id']
        
        # Step 4: Wait for order processing (async)
        import time
        time.sleep(5)
        
        # Step 5: Verify order status
        order_status_response = requests.get(
            f"{services['order_service']}/orders/{order_id}"
        )
        assert order_status_response.status_code == 200
        updated_order = order_status_response.json()
        assert updated_order['status'] == 'confirmed'
        
        # Step 6: Verify inventory was updated
        inventory_response = requests.get(
            f"{services['inventory_service']}/products/PROD123"
        )
        assert inventory_response.status_code == 200
        updated_inventory = inventory_response.json()
        assert updated_inventory['quantity'] == 98  # 100 - 2
        
        # Step 7: Verify user order history
        user_orders_response = requests.get(
            f"{services['user_service']}/users/{user_id}/orders"
        )
        assert user_orders_response.status_code == 200
        user_orders = user_orders_response.json()
        assert len(user_orders) == 1
        assert user_orders[0]['id'] == order_id
    
    def test_order_failure_scenarios(self, services):
        """Test error handling in order flow."""
        
        # Test 1: Order with insufficient inventory
        order_data = {
            'user_id': 999,  # Non-existent user
            'items': [
                {
                    'product_id': 'PROD123',
                    'quantity': 1000  # More than available
                }
            ]
        }
        
        order_response = requests.post(
            f"{services['order_service']}/orders",
            json=order_data
        )
        assert order_response.status_code == 400
        error = order_response.json()
        assert 'insufficient inventory' in error['message'].lower()
        
        # Test 2: Order for non-existent product
        order_data = {
            'user_id': 1,
            'items': [
                {
                    'product_id': 'NONEXISTENT',
                    'quantity': 1
                }
            ]
        }
        
        order_response = requests.post(
            f"{services['order_service']}/orders",
            json=order_data
        )
        assert order_response.status_code == 404
```

### UI End-to-End Tests with Selenium

```python
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

class TestUIE2E:
    
    @pytest.fixture
    def browser(self):
        """Set up Chrome browser for testing."""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Run in headless mode
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.implicitly_wait(10)
        
        yield driver
        driver.quit()
    
    def test_user_registration_and_order(self, browser):
        """Test complete user journey through the UI."""
        
        # Navigate to registration page
        browser.get("http://localhost:3000/register")
        
        # Fill registration form
        browser.find_element(By.ID, "name").send_keys("UI Test User")
        browser.find_element(By.ID, "email").send_keys("ui@example.com")
        browser.find_element(By.ID, "password").send_keys("testpassword")
        browser.find_element(By.ID, "register-btn").click()
        
        # Wait for redirect to dashboard
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.ID, "dashboard"))
        )
        
        # Navigate to products page
        browser.find_element(By.LINK_TEXT, "Products").click()
        
        # Add product to cart
        browser.find_element(By.CSS_SELECTOR, "[data-product-id='PROD123'] .add-to-cart").click()
        
        # Wait for cart update
        WebDriverWait(browser, 5).until(
            EC.text_to_be_present_in_element((By.ID, "cart-count"), "1")
        )
        
        # Go to cart
        browser.find_element(By.ID, "cart-icon").click()
        
        # Proceed to checkout
        browser.find_element(By.ID, "checkout-btn").click()
        
        # Fill shipping information
        browser.find_element(By.ID, "shipping-address").send_keys("123 Test St, Test City")
        browser.find_element(By.ID, "place-order-btn").click()
        
        # Wait for order confirmation
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "order-confirmation"))
        )
        
        # Verify order was placed
        confirmation_text = browser.find_element(By.CLASS_NAME, "order-confirmation").text
        assert "Order placed successfully" in confirmation_text
        
        # Extract order ID for verification
        order_id = browser.find_element(By.ID, "order-id").text
        assert order_id.startswith("ORD")
```

## 5. Test Automation and CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/microservices-tests.yml
name: Microservices Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: [user-service, order-service, inventory-service]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        cd services/${{ matrix.service }}
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run unit tests
      run: |
        cd services/${{ matrix.service }}
        pytest tests/unit/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./services/${{ matrix.service }}/coverage.xml

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: testdb
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:6
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run integration tests
      env:
        DATABASE_URL: postgresql://postgres:testpass@localhost:5432/testdb
        REDIS_URL: redis://localhost:6379
      run: |
        pytest tests/integration/ -v

  contract-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install pact-python
        pip install -r requirements-test.txt
    
    - name: Run consumer contract tests
      run: |
        pytest tests/contract/consumer/ -v
    
    - name: Publish consumer contracts
      if: github.ref == 'refs/heads/main'
      run: |
        python scripts/publish_contracts.py
      env:
        PACT_BROKER_URL: ${{ secrets.PACT_BROKER_URL }}
        PACT_BROKER_TOKEN: ${{ secrets.PACT_BROKER_TOKEN }}
    
    - name: Run provider contract verification
      run: |
        pytest tests/contract/provider/ -v

  e2e-tests:
    runs-on: ubuntu-latest
    needs: [integration-tests, contract-tests]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Start services with docker-compose
      run: |
        docker-compose -f docker-compose.test.yml up -d
        sleep 30  # Wait for services to start
    
    - name: Health check
      run: |
        curl -f http://localhost:8080/health
        curl -f http://localhost:8081/health
        curl -f http://localhost:8082/health
    
    - name: Run E2E tests
      run: |
        pytest tests/e2e/ -v
    
    - name: Collect logs on failure
      if: failure()
      run: |
        docker-compose -f docker-compose.test.yml logs
    
    - name: Cleanup
      if: always()
      run: |
        docker-compose -f docker-compose.test.yml down -v
```

### Test Data Management

```python
# tests/fixtures/test_data.py
import pytest
import asyncio
from datetime import datetime, timedelta
from src.models import User, Product, Order
from src.database import get_database

class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_user(**kwargs):
        """Create a test user with optional overrides."""
        defaults = {
            'name': 'Test User',
            'email': f'test_{int(datetime.now().timestamp())}@example.com',
            'age': 25,
            'created_at': datetime.now()
        }
        defaults.update(kwargs)
        return User(**defaults)
    
    @staticmethod
    def create_product(**kwargs):
        """Create a test product with optional overrides."""
        defaults = {
            'id': f'PROD_{int(datetime.now().timestamp())}',
            'name': 'Test Product',
            'price': 19.99,
            'quantity': 100,
            'category': 'Electronics'
        }
        defaults.update(kwargs)
        return Product(**defaults)
    
    @staticmethod
    def create_order(user_id=None, product_ids=None, **kwargs):
        """Create a test order with optional overrides."""
        defaults = {
            'user_id': user_id or 1,
            'status': 'pending',
            'total': 29.99,
            'created_at': datetime.now()
        }
        defaults.update(kwargs)
        
        order = Order(**defaults)
        if product_ids:
            order.items = [
                {'product_id': pid, 'quantity': 1, 'price': 19.99}
                for pid in product_ids
            ]
        return order

@pytest.fixture
async def clean_database():
    """Clean database before and after each test."""
    db = get_database()
    
    # Clean before test
    await db.execute("TRUNCATE TABLE orders, users, products RESTART IDENTITY CASCADE")
    
    yield db
    
    # Clean after test
    await db.execute("TRUNCATE TABLE orders, users, products RESTART IDENTITY CASCADE")

@pytest.fixture
async def sample_user(clean_database):
    """Create a sample user for testing."""
    user = TestDataFactory.create_user(
        name="Sample User",
        email="sample@example.com"
    )
    
    from src.repositories import UserRepository
    repo = UserRepository(clean_database)
    return await repo.create(user)

@pytest.fixture
async def sample_products(clean_database):
    """Create sample products for testing."""
    products = [
        TestDataFactory.create_product(
            id="PROD1",
            name="Laptop",
            price=999.99,
            quantity=10
        ),
        TestDataFactory.create_product(
            id="PROD2",
            name="Mouse",
            price=29.99,
            quantity=50
        )
    ]
    
    from src.repositories import ProductRepository
    repo = ProductRepository(clean_database)
    created_products = []
    
    for product in products:
        created_products.append(await repo.create(product))
    
    return created_products
```

## 6. Performance Testing

### Load Testing with Locust

```python
# tests/performance/locustfile.py
from locust import HttpUser, task, between
import random
import json

class UserServiceLoadTest(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Set up test user on start."""
        self.user_id = self.create_test_user()
    
    def create_test_user(self):
        """Create a test user and return user ID."""
        user_data = {
            'name': f'LoadTest User {random.randint(1000, 9999)}',
            'email': f'loadtest{random.randint(1000, 9999)}@example.com',
            'age': random.randint(18, 65)
        }
        
        response = self.client.post('/users', json=user_data)
        if response.status_code == 201:
            return response.json()['id']
        return None
    
    @task(3)
    def get_user_profile(self):
        """Test getting user profile - most common operation."""
        if self.user_id:
            self.client.get(f'/users/{self.user_id}')
    
    @task(2)
    def update_user_profile(self):
        """Test updating user profile."""
        if self.user_id:
            update_data = {
                'name': f'Updated User {random.randint(1000, 9999)}'
            }
            self.client.put(f'/users/{self.user_id}', json=update_data)
    
    @task(1)
    def get_user_orders(self):
        """Test getting user orders."""
        if self.user_id:
            self.client.get(f'/users/{self.user_id}/orders')

class OrderServiceLoadTest(HttpUser):
    wait_time = between(2, 5)
    
    def on_start(self):
        """Set up test data."""
        self.user_id = 1  # Assume test user exists
        self.product_ids = ['PROD1', 'PROD2']  # Assume test products exist
    
    @task(5)
    def browse_orders(self):
        """Test browsing orders."""
        self.client.get('/orders')
    
    @task(3)
    def get_order_details(self):
        """Test getting order details."""
        order_id = random.randint(1, 100)
        self.client.get(f'/orders/{order_id}')
    
    @task(1)
    def place_order(self):
        """Test placing a new order."""
        order_data = {
            'user_id': self.user_id,
            'items': [
                {
                    'product_id': random.choice(self.product_ids),
                    'quantity': random.randint(1, 3)
                }
            ]
        }
        self.client.post('/orders', json=order_data)

# Run with: locust -f tests/performance/locustfile.py --host=http://localhost:8080
```

### Stress Testing Script

```python
# tests/performance/stress_test.py
import asyncio
import aiohttp
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

class StressTest:
    def __init__(self, base_url, max_concurrent=100):
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.results = []
    
    async def make_request(self, session, endpoint, method='GET', data=None):
        """Make a single HTTP request and measure response time."""
        start_time = time.time()
        
        try:
            if method == 'GET':
                async with session.get(f"{self.base_url}{endpoint}") as response:
                    await response.text()
                    status = response.status
            elif method == 'POST':
                async with session.post(f"{self.base_url}{endpoint}", json=data) as response:
                    await response.text()
                    status = response.status
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            return {
                'endpoint': endpoint,
                'method': method,
                'status': status,
                'response_time': response_time,
                'success': 200 <= status < 300
            }
            
        except Exception as e:
            end_time = time.time()
            return {
                'endpoint': endpoint,
                'method': method,
                'status': 0,
                'response_time': (end_time - start_time) * 1000,
                'success': False,
                'error': str(e)
            }
    
    async def run_concurrent_requests(self, endpoint, num_requests, method='GET', data=None):
        """Run multiple concurrent requests to an endpoint."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for _ in range(num_requests):
                task = self.make_request(session, endpoint, method, data)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
    
    def analyze_results(self, results):
        """Analyze stress test results."""
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        if successful_requests:
            response_times = [r['response_time'] for r in successful_requests]
            
            analysis = {
                'total_requests': len(results),
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'success_rate': len(successful_requests) / len(results) * 100,
                'average_response_time': statistics.mean(response_times),
                'median_response_time': statistics.median(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times),
                'p95_response_time': self.percentile(response_times, 95),
                'p99_response_time': self.percentile(response_times, 99)
            }
        else:
            analysis = {
                'total_requests': len(results),
                'successful_requests': 0,
                'failed_requests': len(failed_requests),
                'success_rate': 0
            }
        
        return analysis
    
    def percentile(self, data, percentile):
        """Calculate percentile of a dataset."""
        size = len(data)
        return sorted(data)[int(size * percentile / 100)]
    
    async def run_stress_test(self):
        """Run comprehensive stress test."""
        test_scenarios = [
            {'endpoint': '/users/1', 'requests': 1000, 'method': 'GET'},
            {'endpoint': '/orders', 'requests': 500, 'method': 'GET'},
            {
                'endpoint': '/orders',
                'requests': 100,
                'method': 'POST',
                'data': {
                    'user_id': 1,
                    'items': [{'product_id': 'PROD1', 'quantity': 1}]
                }
            }
        ]
        
        print("Starting stress test...")
        
        for scenario in test_scenarios:
            print(f"\nTesting {scenario['method']} {scenario['endpoint']} "
                  f"with {scenario['requests']} requests...")
            
            results = await self.run_concurrent_requests(
                scenario['endpoint'],
                scenario['requests'],
                scenario['method'],
                scenario.get('data')
            )
            
            analysis = self.analyze_results(results)
            
            print(f"Results for {scenario['endpoint']}:")
            print(f"  Success Rate: {analysis['success_rate']:.2f}%")
            if analysis['successful_requests'] > 0:
                print(f"  Average Response Time: {analysis['average_response_time']:.2f}ms")
                print(f"  95th Percentile: {analysis['p95_response_time']:.2f}ms")
                print(f"  99th Percentile: {analysis['p99_response_time']:.2f}ms")
            
            # Store results for overall analysis
            self.results.extend(results)

if __name__ == '__main__':
    stress_test = StressTest('http://localhost:8080')
    asyncio.run(stress_test.run_stress_test())
```

## 7. Security Testing

### Security Test Suite

```python
# tests/security/security_tests.py
import pytest
import requests
import jwt
from datetime import datetime, timedelta

class TestSecurityVulnerabilities:
    
    def test_sql_injection_protection(self):
        """Test protection against SQL injection attacks."""
        malicious_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "'; INSERT INTO users (name) VALUES ('hacker'); --"
        ]
        
        for payload in malicious_payloads:
            response = requests.get(
                'http://localhost:8080/users',
                params={'name': payload}
            )
            
            # Should not return 500 error or expose database structure
            assert response.status_code != 500
            assert 'database' not in response.text.lower()
            assert 'sql' not in response.text.lower()
    
    def test_xss_protection(self):
        """Test protection against XSS attacks."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "'; alert('XSS'); //"
        ]
        
        for payload in xss_payloads:
            user_data = {
                'name': payload,
                'email': 'test@example.com'
            }
            
            response = requests.post(
                'http://localhost:8080/users',
                json=user_data
            )
            
            # Check that script tags are escaped or removed
            if response.status_code == 201:
                user = response.json()
                assert '<script>' not in user['name']
                assert 'javascript:' not in user['name']
    
    def test_authentication_required(self):
        """Test that protected endpoints require authentication."""
        protected_endpoints = [
            'GET /users/profile',
            'PUT /users/1',
            'DELETE /users/1',
            'POST /orders',
            'GET /admin/users'
        ]
        
        for endpoint in protected_endpoints:
            method, path = endpoint.split(' ')
            
            if method == 'GET':
                response = requests.get(f'http://localhost:8080{path}')
            elif method == 'POST':
                response = requests.post(f'http://localhost:8080{path}', json={})
            elif method == 'PUT':
                response = requests.put(f'http://localhost:8080{path}', json={})
            elif method == 'DELETE':
                response = requests.delete(f'http://localhost:8080{path}')
            
            assert response.status_code == 401, f"Endpoint {endpoint} should require authentication"
    
    def test_jwt_token_validation(self):
        """Test JWT token validation."""
        # Test with invalid token
        invalid_token = "invalid.jwt.token"
        headers = {'Authorization': f'Bearer {invalid_token}'}
        
        response = requests.get(
            'http://localhost:8080/users/profile',
            headers=headers
        )
        assert response.status_code == 401
        
        # Test with expired token
        expired_payload = {
            'user_id': 1,
            'exp': datetime.utcnow() - timedelta(hours=1)
        }
        expired_token = jwt.encode(expired_payload, 'secret', algorithm='HS256')
        headers = {'Authorization': f'Bearer {expired_token}'}
        
        response = requests.get(
            'http://localhost:8080/users/profile',
            headers=headers
        )
        assert response.status_code == 401
    
    def test_rate_limiting(self):
        """Test rate limiting protection."""
        # Make rapid requests to test rate limiting
        responses = []
        
        for i in range(50):  # Assuming rate limit is less than 50 requests
            response = requests.get('http://localhost:8080/users')
            responses.append(response.status_code)
        
        # Should eventually get 429 (Too Many Requests)
        assert 429 in responses, "Rate limiting should be enforced"
    
    def test_input_validation(self):
        """Test input validation and sanitization."""
        # Test with invalid email format
        invalid_user_data = [
            {'name': '', 'email': 'invalid-email'},  # Invalid email
            {'name': 'A' * 1000, 'email': 'test@example.com'},  # Name too long
            {'name': 'Test', 'email': ''},  # Empty email
            {'name': 'Test', 'age': -5},  # Invalid age
            {'name': 'Test', 'age': 'not_a_number'}  # Non-numeric age
        ]
        
        for user_data in invalid_user_data:
            response = requests.post(
                'http://localhost:8080/users',
                json=user_data
            )
            
            assert response.status_code == 400, f"Should reject invalid data: {user_data}"
    
    def test_cors_headers(self):
        """Test CORS headers configuration."""
        response = requests.options('http://localhost:8080/users')
        
        # Check for proper CORS headers
        assert 'Access-Control-Allow-Origin' in response.headers
        assert 'Access-Control-Allow-Methods' in response.headers
        assert 'Access-Control-Allow-Headers' in response.headers
    
    def test_sensitive_data_exposure(self):
        """Test that sensitive data is not exposed."""
        # Create user with password
        user_data = {
            'name': 'Test User',
            'email': 'test@example.com',
            'password': 'secret123'
        }
        
        response = requests.post('http://localhost:8080/users', json=user_data)
        
        if response.status_code == 201:
            user = response.json()
            
            # Password should not be in response
            assert 'password' not in user
            assert 'secret123' not in str(user)
            
            # Get user endpoint should also not expose password
            user_response = requests.get(f'http://localhost:8080/users/{user["id"]}')
            if user_response.status_code == 200:
                user_details = user_response.json()
                assert 'password' not in user_details
```

## 8. Test Monitoring and Reporting

### Test Metrics Collection

```python
# tests/monitoring/test_metrics.py
import pytest
import time
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class TestMetric:
    test_name: str
    test_type: str  # unit, integration, e2e, contract
    duration: float
    status: str  # passed, failed, skipped
    timestamp: datetime
    service: str
    error_message: str = None

class TestMetricsCollector:
    def __init__(self):
        self.metrics: List[TestMetric] = []
    
    def record_test(self, test_name: str, test_type: str, service: str, 
                   duration: float, status: str, error_message: str = None):
        """Record test execution metrics."""
        metric = TestMetric(
            test_name=test_name,
            test_type=test_type,
            duration=duration,
            status=status,
            timestamp=datetime.now(),
            service=service,
            error_message=error_message
        )
        self.metrics.append(metric)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        if not self.metrics:
            return {}
        
        total_tests = len(self.metrics)
        passed_tests = len([m for m in self.metrics if m.status == 'passed'])
        failed_tests = len([m for m in self.metrics if m.status == 'failed'])
        skipped_tests = len([m for m in self.metrics if m.status == 'skipped'])
        
        # Calculate metrics by test type
        type_metrics = {}
        for test_type in ['unit', 'integration', 'e2e', 'contract']:
            type_tests = [m for m in self.metrics if m.test_type == test_type]
            if type_tests:
                type_metrics[test_type] = {
                    'total': len(type_tests),
                    'passed': len([m for m in type_tests if m.status == 'passed']),
                    'failed': len([m for m in type_tests if m.status == 'failed']),
                    'average_duration': sum(m.duration for m in type_tests) / len(type_tests)
                }
        
        # Calculate metrics by service
        service_metrics = {}
        services = set(m.service for m in self.metrics)
        for service in services:
            service_tests = [m for m in self.metrics if m.service == service]
            service_metrics[service] = {
                'total': len(service_tests),
                'passed': len([m for m in service_tests if m.status == 'passed']),
                'failed': len([m for m in service_tests if m.status == 'failed']),
                'coverage': len([m for m in service_tests if m.status == 'passed']) / len(service_tests) * 100
            }
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'skipped': skipped_tests,
                'success_rate': passed_tests / total_tests * 100 if total_tests > 0 else 0,
                'total_duration': sum(m.duration for m in self.metrics),
                'average_duration': sum(m.duration for m in self.metrics) / total_tests
            },
            'by_type': type_metrics,
            'by_service': service_metrics,
            'failed_tests': [
                {
                    'name': m.test_name,
                    'service': m.service,
                    'type': m.test_type,
                    'error': m.error_message
                }
                for m in self.metrics if m.status == 'failed'
            ]
        }
    
    def export_to_json(self, filename: str):
        """Export metrics to JSON file."""
        report = self.generate_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

# Pytest plugin for metrics collection
@pytest.fixture(scope="session")
def metrics_collector():
    return TestMetricsCollector()

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    start_time = time.time()
    outcome = yield
    end_time = time.time()
    
    duration = end_time - start_time
    
    # Determine test type based on file path
    test_file = str(item.fspath)
    if 'unit' in test_file:
        test_type = 'unit'
    elif 'integration' in test_file:
        test_type = 'integration'
    elif 'e2e' in test_file:
        test_type = 'e2e'
    elif 'contract' in test_file:
        test_type = 'contract'
    else:
        test_type = 'unknown'
    
    # Determine service from test path
    service = 'unknown'
    if 'user-service' in test_file or 'user_service' in test_file:
        service = 'user-service'
    elif 'order-service' in test_file or 'order_service' in test_file:
        service = 'order-service'
    elif 'inventory-service' in test_file or 'inventory_service' in test_file:
        service = 'inventory-service'
    
    # Determine test status
    if outcome.excinfo is None:
        status = 'passed'
        error_message = None
    else:
        status = 'failed'
        error_message = str(outcome.excinfo[1]) if outcome.excinfo else None
    
    # Record metrics (if collector is available)
    if hasattr(item.session, 'metrics_collector'):
        item.session.metrics_collector.record_test(
            test_name=item.name,
            test_type=test_type,
            service=service,
            duration=duration,
            status=status,
            error_message=error_message
        )
```

## 9. Best Practices and Guidelines

### Testing Best Practices

#### 1. Test Structure and Organization

```python
# Good: Clear test structure following AAA pattern
def test_user_creation_with_valid_data():
    # Arrange
    user_data = {
        'name': 'John Doe',
        'email': 'john@example.com',
        'age': 30
    }
    
    # Act
    user = user_service.create_user(user_data)
    
    # Assert
    assert user.name == 'John Doe'
    assert user.email == 'john@example.com'
    assert user.id is not None

# Bad: Unclear test without proper structure
def test_user():
    user = user_service.create_user({'name': 'John', 'email': 'john@example.com'})
    assert user.name == 'John'
    user2 = user_service.create_user({'name': 'Jane', 'email': 'jane@example.com'})
    assert len(user_service.get_all_users()) == 2
```

#### 2. Test Naming Conventions

```python
# Good: Descriptive test names
def test_create_user_with_duplicate_email_raises_validation_error():
    pass

def test_get_user_by_id_returns_user_when_exists():
    pass

def test_delete_user_removes_user_from_database():
    pass

# Bad: Unclear test names
def test_user_creation():
    pass

def test_get_user():
    pass

def test_delete():
    pass
```

#### 3. Test Data Management

```python
# Good: Use factories and fixtures for test data
@pytest.fixture
def valid_user_data():
    return {
        'name': 'Test User',
        'email': f'test_{uuid4()}@example.com',
        'age': 25
    }

def test_user_creation(valid_user_data):
    user = user_service.create_user(valid_user_data)
    assert user.email == valid_user_data['email']

# Bad: Hardcoded test data that can cause conflicts
def test_user_creation():
    user_data = {
        'name': 'John Doe',
        'email': 'john@example.com'  # This might already exist!
    }
    user = user_service.create_user(user_data)
```

#### 4. Mock Usage Guidelines

```python
# Good: Mock external dependencies, test your code
@patch('src.user_service.email_service')
def test_user_creation_sends_welcome_email(mock_email_service):
    user_data = {'name': 'John', 'email': 'john@example.com'}
    
    user_service.create_user(user_data)
    
    mock_email_service.send_welcome_email.assert_called_once_with('john@example.com')

# Bad: Over-mocking internal logic
@patch('src.user_service.UserService.validate_email')
@patch('src.user_service.UserService.hash_password')
def test_user_creation(mock_hash, mock_validate):
    # This mocks too much of the actual implementation
    pass
```

### Common Pitfalls to Avoid

#### 1. Test Dependencies
```python
# Bad: Tests that depend on each other
class TestUserService:
    def test_create_user(self):
        user = user_service.create_user({'name': 'John'})
        self.created_user_id = user.id  # Storing state!
    
    def test_get_user(self):
        user = user_service.get_user(self.created_user_id)  # Depends on previous test!
        assert user.name == 'John'

# Good: Independent tests
class TestUserService:
    def test_create_user(self):
        user = user_service.create_user({'name': 'John'})
        assert user.name == 'John'
    
    def test_get_user(self, sample_user):  # Use fixture
        user = user_service.get_user(sample_user.id)
        assert user.name == sample_user.name
```

#### 2. Flaky Tests
```python
# Bad: Time-dependent test
def test_user_creation_timestamp():
    user = user_service.create_user({'name': 'John'})
    assert user.created_at == datetime.now()  # This will fail due to timing!

# Good: Use time mocking or ranges
@patch('src.user_service.datetime')
def test_user_creation_timestamp(mock_datetime):
    fixed_time = datetime(2025, 1, 1, 12, 0, 0)
    mock_datetime.now.return_value = fixed_time
    
    user = user_service.create_user({'name': 'John'})
    assert user.created_at == fixed_time
```

#### 3. Testing Implementation Details
```python
# Bad: Testing internal implementation
def test_user_service_calls_repository():
    with patch('src.user_service.UserRepository') as mock_repo:
        user_service.create_user({'name': 'John'})
        mock_repo.save.assert_called_once()  # Testing internal calls

# Good: Testing behavior and outcomes
def test_create_user_persists_user():
    user_data = {'name': 'John', 'email': 'john@example.com'}
    
    created_user = user_service.create_user(user_data)
    retrieved_user = user_service.get_user(created_user.id)
    
    assert retrieved_user.name == 'John'
    assert retrieved_user.email == 'john@example.com'
```

### Test Maintenance

#### 1. Regular Test Review
- Review test coverage regularly
- Remove obsolete tests
- Update tests when requirements change
- Refactor test code like production code

#### 2. Performance Optimization
```python
# Optimize slow tests
@pytest.mark.slow
def test_large_dataset_processing():
    # Mark slow tests to run separately
    pass

# Use smaller datasets for unit tests
def test_user_filtering():
    users = [create_test_user() for _ in range(10)]  # Not 10,000!
    filtered = user_service.filter_users(users, age_min=18)
    assert len(filtered) > 0
```

#### 3. Documentation
```python
def test_complex_business_logic():
    """
    Test the complex discount calculation logic.
    
    Business rules:
    - Regular customers get 5% discount
    - Premium customers get 10% discount  
    - Orders over $100 get additional 5% discount
    - Maximum discount is 20%
    """
    # Test implementation
    pass
```

## Learning Objectives

By the end of this section, you should be able to:

- **Design comprehensive test strategies** for distributed microservices systems
- **Implement unit tests** with proper mocking and isolation techniques
- **Create integration tests** that verify service interactions
- **Develop contract tests** to ensure API compatibility between services
- **Build end-to-end test suites** that validate complete user workflows
- **Set up automated testing pipelines** in CI/CD environments
- **Conduct performance and security testing** for microservices
- **Monitor and analyze test metrics** to improve testing effectiveness
- **Apply testing best practices** to maintain high-quality test suites

### Self-Assessment Checklist

Before proceeding to the next topic, ensure you can:

□ Write unit tests with >90% code coverage for a microservice  
□ Create integration tests using test containers for databases  
□ Implement consumer and provider contract tests using Pact  
□ Design and execute end-to-end test scenarios  
□ Set up automated test execution in CI/CD pipelines  
□ Perform load testing and interpret performance metrics  
□ Conduct security testing for common vulnerabilities  
□ Implement test data management strategies  
□ Use proper mocking techniques without over-mocking  
□ Organize tests following best practices and conventions  

### Practical Exercises

**Exercise 1: Complete Test Suite**
Create a comprehensive test suite for a simple e-commerce microservice including:
- Unit tests with >90% coverage
- Integration tests with real database
- Contract tests for external APIs
- Basic load testing

**Exercise 2: Test Automation Pipeline**
Set up a complete CI/CD pipeline that:
- Runs tests in parallel
- Generates coverage reports
- Performs security scanning
- Deploys only if all tests pass

**Exercise 3: Performance Testing**
Design and implement performance tests that:
- Test different load patterns
- Identify performance bottlenecks
- Generate actionable reports
- Set up performance monitoring

## Study Materials

### Recommended Reading
- **Primary:** "Testing Microservices with Mocha and Chai" by Daniel Li
- **Reference:** "Microservices Testing (Live Book)" by Toby Clemson
- **Online:** [Martin Fowler's Testing Strategies](https://martinfowler.com/articles/microservice-testing/)
- **Documentation:** [Pytest Documentation](https://docs.pytest.org/), [Pact Documentation](https://docs.pact.io/)

### Tools and Frameworks
- **Unit Testing:** pytest, unittest, Jest, JUnit
- **Contract Testing:** Pact, Spring Cloud Contract
- **Load Testing:** Locust, Artillery, JMeter, Gatling
- **Security Testing:** OWASP ZAP, Bandit, Safety
- **Test Containers:** Testcontainers, Docker Compose
- **CI/CD:** GitHub Actions, Jenkins, GitLab CI

### Practice Projects
1. **Microservices Test Suite:** Build comprehensive tests for a multi-service application
2. **Contract Testing Implementation:** Implement Pact contract tests between services
3. **Performance Testing Framework:** Create reusable performance testing infrastructure
4. **Security Testing Automation:** Implement automated security testing in CI/CD

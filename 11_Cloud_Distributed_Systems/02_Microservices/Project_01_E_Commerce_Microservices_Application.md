# Project 1: E-Commerce Microservices Application

*Duration: 3-4 weeks | Difficulty: Intermediate to Advanced*

## Project Overview

Build a production-ready e-commerce system using microservices architecture. This project demonstrates key microservices patterns including service discovery, API gateway, inter-service communication, distributed data management, and fault tolerance.

### What You'll Learn
- **Microservices Architecture**: Design and implement loosely coupled services
- **Service Discovery**: Automatic service registration and discovery
- **API Gateway Pattern**: Centralized entry point with routing and cross-cutting concerns
- **Inter-Service Communication**: Synchronous (REST) and asynchronous (message queues) communication
- **Data Management**: Database per service pattern and eventual consistency
- **Containerization**: Docker and Docker Compose for service orchestration
- **Monitoring & Logging**: Distributed tracing and centralized logging
- **Testing Strategies**: Unit, integration, and contract testing for microservices

### Business Requirements
Our e-commerce platform needs to handle:
- **Product Management**: Catalog browsing, search, inventory management
- **User Management**: Registration, authentication, user profiles
- **Shopping Cart**: Add/remove items, cart persistence
- **Order Processing**: Order creation, payment processing, order tracking
- **Notifications**: Email/SMS notifications for order updates
- **Analytics**: User behavior tracking and business metrics

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │  Mobile App     │    │  Admin Panel    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │      API Gateway          │
                    │   (Load Balancer,         │
                    │   Authentication,         │
                    │   Rate Limiting)          │
                    └─────────┬─────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    ┌─────▼─────┐    ┌────────▼────────┐    ┌─────▼─────┐
    │  Product  │    │      User       │    │   Order   │
    │  Service  │    │    Service      │    │  Service  │
    │           │    │                 │    │           │
    │ PostgreSQL│    │    PostgreSQL   │    │ PostgreSQL│
    └─────┬─────┘    └────────┬────────┘    └─────┬─────┘
          │                   │                   │
    ┌─────▼─────┐    ┌────────▼────────┐    ┌─────▼─────┐
    │   Cart    │    │    Payment      │    │Notification│
    │  Service  │    │    Service      │    │  Service  │
    │           │    │                 │    │           │
    │   Redis   │    │    PostgreSQL   │    │  MongoDB  │
    └─────┬─────┘    └────────┬────────┘    └─────┬─────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
                    ┌─────────▼─────────────┐
                    │   Message Queue       │
                    │     (RabbitMQ)        │
                    └───────────────────────┘
```

## Implementation Guide

### Phase 1: Core Services Development

#### 1. Product Service
**Responsibilities**: Product catalog management, search, inventory tracking

**Database Schema:**
```sql
-- products.sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    category_id INTEGER,
    stock_quantity INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES categories(id)
);

CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_products_name ON products(name);
```

**Complete Product Service Implementation:**
```python
# product_service/app.py
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os
import logging
from datetime import datetime
import requests

app = Flask(__name__)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'DATABASE_URL', 
    'postgresql://user:password@localhost/products_db'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models
class Category(db.Model):
    __tablename__ = 'categories'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey('categories.id'))
    
    products = db.relationship('Product', backref='category', lazy=True)
    children = db.relationship('Category', backref=db.backref('parent', remote_side=[id]))
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'parent_id': self.parent_id
        }

class Product(db.Model):
    __tablename__ = 'products'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    price = db.Column(db.Numeric(10, 2), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('categories.id'))
    stock_quantity = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'price': float(self.price),
            'category_id': self.category_id,
            'stock_quantity': self.stock_quantity,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

# Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for service discovery"""
    return jsonify({'status': 'healthy', 'service': 'product-service'})

@app.route('/products', methods=['GET'])
def get_products():
    """Get all products with optional filtering"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        category_id = request.args.get('category_id', type=int)
        search = request.args.get('search', '')
        
        query = Product.query
        
        if category_id:
            query = query.filter(Product.category_id == category_id)
        
        if search:
            query = query.filter(Product.name.ilike(f'%{search}%'))
        
        products = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        return jsonify({
            'products': [product.to_dict() for product in products.items],
            'pagination': {
                'page': page,
                'pages': products.pages,
                'per_page': per_page,
                'total': products.total
            }
        })
    except Exception as e:
        logger.error(f"Error fetching products: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/products/<int:product_id>', methods=['GET'])
def get_product(product_id):
    """Get a specific product by ID"""
    try:
        product = Product.query.get_or_404(product_id)
        return jsonify(product.to_dict())
    except Exception as e:
        logger.error(f"Error fetching product {product_id}: {str(e)}")
        return jsonify({'error': 'Product not found'}), 404

@app.route('/products', methods=['POST'])
def create_product():
    """Create a new product"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'price']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        product = Product(
            name=data['name'],
            description=data.get('description'),
            price=data['price'],
            category_id=data.get('category_id'),
            stock_quantity=data.get('stock_quantity', 0)
        )
        
        db.session.add(product)
        db.session.commit()
        
        logger.info(f"Created product: {product.id}")
        return jsonify(product.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating product: {str(e)}")
        return jsonify({'error': 'Failed to create product'}), 500

@app.route('/products/<int:product_id>/stock', methods=['PUT'])
def update_stock(product_id):
    """Update product stock quantity"""
    try:
        product = Product.query.get_or_404(product_id)
        data = request.get_json()
        
        if 'quantity' not in data:
            return jsonify({'error': 'Missing quantity field'}), 400
        
        product.stock_quantity = data['quantity']
        product.updated_at = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"Updated stock for product {product_id}: {data['quantity']}")
        return jsonify(product.to_dict())
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating stock for product {product_id}: {str(e)}")
        return jsonify({'error': 'Failed to update stock'}), 500

@app.route('/products/<int:product_id>/reserve', methods=['POST'])
def reserve_stock(product_id):
    """Reserve stock for an order (called by order service)"""
    try:
        product = Product.query.get_or_404(product_id)
        data = request.get_json()
        quantity = data.get('quantity', 0)
        
        if product.stock_quantity < quantity:
            return jsonify({'error': 'Insufficient stock'}), 400
        
        product.stock_quantity -= quantity
        product.updated_at = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"Reserved {quantity} units of product {product_id}")
        return jsonify({'message': 'Stock reserved', 'remaining_stock': product.stock_quantity})
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error reserving stock for product {product_id}: {str(e)}")
        return jsonify({'error': 'Failed to reserve stock'}), 500

# Categories endpoints
@app.route('/categories', methods=['GET'])
def get_categories():
    """Get all categories"""
    try:
        categories = Category.query.all()
        return jsonify([category.to_dict() for category in categories])
    except Exception as e:
        logger.error(f"Error fetching categories: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5001)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )
```

**Product Service Configuration:**
```python
# product_service/requirements.txt
Flask==2.3.3
Flask-SQLAlchemy==3.0.5
Flask-Migrate==4.0.5
psycopg2-binary==2.9.7
python-dotenv==1.0.0
requests==2.31.0
gunicorn==21.2.0
```

```dockerfile
# product_service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "4", "app:app"]
```

#### 4. API Gateway Implementation
**Responsibilities**: Request routing, authentication, rate limiting, load balancing

**Complete API Gateway Implementation:**
```python
# api_gateway/app.py
from flask import Flask, request, jsonify, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_jwt_extended import JWTManager, verify_jwt_in_request, get_jwt_identity
import requests
import os
import logging
import time
import json
from functools import wraps
import random
from urllib.parse import urljoin

app = Flask(__name__)

# Configuration
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')

jwt = JWTManager(app)

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service registry
SERVICES = {
    'user': {
        'urls': [
            os.getenv('USER_SERVICE_URL', 'http://localhost:5002'),
        ],
        'health_path': '/health',
        'timeout': 5
    },
    'product': {
        'urls': [
            os.getenv('PRODUCT_SERVICE_URL', 'http://localhost:5001'),
        ],
        'health_path': '/health',
        'timeout': 5
    },
    'cart': {
        'urls': [
            os.getenv('CART_SERVICE_URL', 'http://localhost:5003'),
        ],
        'health_path': '/health',
        'timeout': 5
    },
    'order': {
        'urls': [
            os.getenv('ORDER_SERVICE_URL', 'http://localhost:5004'),
        ],
        'health_path': '/health',
        'timeout': 5
    },
    'payment': {
        'urls': [
            os.getenv('PAYMENT_SERVICE_URL', 'http://localhost:5005'),
        ],
        'health_path': '/health',
        'timeout': 5
    }
}

class ServiceRegistry:
    def __init__(self):
        self.healthy_services = {}
        self.last_health_check = {}
        self.health_check_interval = 30  # seconds
    
    def get_service_url(self, service_name):
        """Get a healthy service URL using round-robin load balancing"""
        if service_name not in SERVICES:
            return None
        
        # Check if we need to refresh health status
        self._refresh_health_status(service_name)
        
        healthy_urls = self.healthy_services.get(service_name, [])
        if not healthy_urls:
            # Fallback to first URL if no healthy services
            return SERVICES[service_name]['urls'][0]
        
        # Simple round-robin
        return random.choice(healthy_urls)
    
    def _refresh_health_status(self, service_name):
        """Refresh health status for a service"""
        now = time.time()
        last_check = self.last_health_check.get(service_name, 0)
        
        if now - last_check < self.health_check_interval:
            return
        
        service_config = SERVICES[service_name]
        healthy_urls = []
        
        for url in service_config['urls']:
            try:
                health_url = urljoin(url, service_config['health_path'])
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200:
                    healthy_urls.append(url)
            except requests.RequestException:
                logger.warning(f"Health check failed for {url}")
        
        self.healthy_services[service_name] = healthy_urls
        self.last_health_check[service_name] = now
        
        if not healthy_urls:
            logger.error(f"No healthy instances for service: {service_name}")

service_registry = ServiceRegistry()

def require_auth(f):
    """Decorator to require JWT authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            verify_jwt_in_request()
            g.current_user_id = get_jwt_identity()
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'error': 'Authentication required'}), 401
    return decorated_function

def proxy_request(service_name, path, method='GET', require_authentication=False):
    """Proxy request to a microservice"""
    try:
        # Get service URL
        service_url = service_registry.get_service_url(service_name)
        if not service_url:
            return jsonify({'error': f'Service {service_name} not available'}), 503
        
        # Construct full URL
        url = urljoin(service_url, path)
        
        # Prepare headers
        headers = dict(request.headers)
        headers.pop('Host', None)  # Remove host header
        
        # Add user context if authenticated
        if require_authentication and hasattr(g, 'current_user_id'):
            headers['X-User-ID'] = str(g.current_user_id)
        
        # Prepare request data
        data = None
        if request.is_json:
            data = request.get_json()
        
        # Make request to service
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=data,
            params=request.args,
            timeout=SERVICES[service_name]['timeout']
        )
        
        # Log request
        logger.info(f"Proxied {method} {url} -> {response.status_code}")
        
        # Return response
        return jsonify(response.json() if response.content else {}), response.status_code
        
    except requests.exceptions.Timeout:
        logger.error(f"Timeout calling {service_name} service")
        return jsonify({'error': 'Service timeout'}), 504
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error calling {service_name} service")
        return jsonify({'error': 'Service unavailable'}), 503
    except Exception as e:
        logger.error(f"Error proxying request to {service_name}: {str(e)}")
        return jsonify({'error': 'Internal gateway error'}), 500

# Health check
@app.route('/health', methods=['GET'])
def health_check():
    """Gateway health check"""
    try:
        service_status = {}
        overall_healthy = True
        
        for service_name in SERVICES.keys():
            service_registry._refresh_health_status(service_name)
            healthy_count = len(service_registry.healthy_services.get(service_name, []))
            total_count = len(SERVICES[service_name]['urls'])
            
            service_status[service_name] = {
                'healthy_instances': healthy_count,
                'total_instances': total_count,
                'status': 'healthy' if healthy_count > 0 else 'unhealthy'
            }
            
            if healthy_count == 0:
                overall_healthy = False
        
        return jsonify({
            'status': 'healthy' if overall_healthy else 'degraded',
            'service': 'api-gateway',
            'services': service_status
        }), 200 if overall_healthy else 503
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

# Authentication routes
@app.route('/api/auth/register', methods=['POST'])
@limiter.limit("5 per minute")
def register():
    """User registration"""
    return proxy_request('user', '/register', 'POST')

@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    """User login"""
    return proxy_request('user', '/login', 'POST')

# User management routes
@app.route('/api/users/profile', methods=['GET'])
@require_auth
@limiter.limit("30 per minute")
def get_profile():
    """Get user profile"""
    return proxy_request('user', '/profile', 'GET', require_authentication=True)

@app.route('/api/users/profile', methods=['PUT'])
@require_auth
@limiter.limit("10 per minute")
def update_profile():
    """Update user profile"""
    return proxy_request('user', '/profile', 'PUT', require_authentication=True)

# Product routes
@app.route('/api/products', methods=['GET'])
@limiter.limit("100 per minute")
def get_products():
    """Get products"""
    return proxy_request('product', '/products', 'GET')

@app.route('/api/products/<int:product_id>', methods=['GET'])
@limiter.limit("100 per minute")
def get_product(product_id):
    """Get specific product"""
    return proxy_request('product', f'/products/{product_id}', 'GET')

@app.route('/api/categories', methods=['GET'])
@limiter.limit("50 per minute")
def get_categories():
    """Get categories"""
    return proxy_request('product', '/categories', 'GET')

# Cart routes
@app.route('/api/cart', methods=['GET'])
@require_auth
@limiter.limit("50 per minute")
def get_cart():
    """Get user's cart"""
    user_id = g.current_user_id
    return proxy_request('cart', f'/cart/{user_id}', 'GET', require_authentication=True)

@app.route('/api/cart/items', methods=['POST'])
@require_auth
@limiter.limit("20 per minute")
def add_to_cart():
    """Add item to cart"""
    user_id = g.current_user_id
    return proxy_request('cart', f'/cart/{user_id}/items', 'POST', require_authentication=True)

@app.route('/api/cart/items/<int:product_id>', methods=['PUT'])
@require_auth
@limiter.limit("20 per minute")
def update_cart_item(product_id):
    """Update cart item"""
    user_id = g.current_user_id
    return proxy_request('cart', f'/cart/{user_id}/items/{product_id}', 'PUT', require_authentication=True)

@app.route('/api/cart/items/<int:product_id>', methods=['DELETE'])
@require_auth
@limiter.limit("20 per minute")
def remove_from_cart(product_id):
    """Remove item from cart"""
    user_id = g.current_user_id
    return proxy_request('cart', f'/cart/{user_id}/items/{product_id}', 'DELETE', require_authentication=True)

# Order routes
@app.route('/api/orders', methods=['GET'])
@require_auth
@limiter.limit("30 per minute")
def get_orders():
    """Get user's orders"""
    return proxy_request('order', '/orders', 'GET', require_authentication=True)

@app.route('/api/orders', methods=['POST'])
@require_auth
@limiter.limit("5 per minute")
def create_order():
    """Create new order"""
    return proxy_request('order', '/orders', 'POST', require_authentication=True)

@app.route('/api/orders/<int:order_id>', methods=['GET'])
@require_auth
@limiter.limit("30 per minute")
def get_order(order_id):
    """Get specific order"""
    return proxy_request('order', f'/orders/{order_id}', 'GET', require_authentication=True)

# Payment routes
@app.route('/api/payments/process', methods=['POST'])
@require_auth
@limiter.limit("3 per minute")
def process_payment():
    """Process payment"""
    return proxy_request('payment', '/payments/process', 'POST', require_authentication=True)

# Error handlers
@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({'error': 'Rate limit exceeded'}), 429

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Request/response logging middleware
@app.before_request
def log_request():
    g.start_time = time.time()
    logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")

@app.after_request
def log_response(response):
    duration = time.time() - g.start_time
    logger.info(f"Response: {response.status_code} in {duration:.3f}s")
    
    # Add CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    
    return response

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 8000)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )
```

**API Gateway Requirements:**
```python
# api_gateway/requirements.txt
Flask==2.3.3
Flask-Limiter==3.5.0
Flask-JWT-Extended==4.5.2
requests==2.31.0
python-dotenv==1.0.0
gunicorn==21.2.0
```

#### 2. User Service
**Responsibilities**: User registration, authentication, profile management, JWT token generation

**Complete User Service Implementation:**
```python
# user_service/app.py
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
import os
import logging
from datetime import datetime, timedelta
import re

app = Flask(__name__)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'DATABASE_URL', 
    'postgresql://user:password@localhost/users_db'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    phone = db.Column(db.String(20))
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    addresses = db.relationship('Address', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)
    
    def to_dict(self, include_sensitive=False):
        result = {
            'id': self.id,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'phone': self.phone,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
        }
        if include_sensitive:
            result['is_admin'] = self.is_admin
        return result

class Address(db.Model):
    __tablename__ = 'addresses'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    street_address = db.Column(db.String(255), nullable=False)
    city = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(100), nullable=False)
    postal_code = db.Column(db.String(20), nullable=False)
    country = db.Column(db.String(100), nullable=False)
    is_default = db.Column(db.Boolean, default=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'street_address': self.street_address,
            'city': self.city,
            'state': self.state,
            'postal_code': self.postal_code,
            'country': self.country,
            'is_default': self.is_default
        }

# Utility functions
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    # At least 8 characters, one uppercase, one lowercase, one digit
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    return True, "Valid password"

# Routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'user-service'})

@app.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['email', 'password', 'first_name', 'last_name']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Missing or empty field: {field}'}), 400
        
        # Validate email format
        if not validate_email(data['email']):
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Validate password strength
        is_valid, message = validate_password(data['password'])
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Check if user already exists
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already registered'}), 409
        
        # Create new user
        user = User(
            email=data['email'].lower().strip(),
            first_name=data['first_name'].strip(),
            last_name=data['last_name'].strip(),
            phone=data.get('phone', '').strip()
        )
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.commit()
        
        logger.info(f"New user registered: {user.email}")
        
        # Generate access token
        access_token = create_access_token(identity=user.id)
        
        return jsonify({
            'message': 'User registered successfully',
            'user': user.to_dict(),
            'access_token': access_token
        }), 201
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error during user registration: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/login', methods=['POST'])
def login():
    """Authenticate user and return JWT token"""
    try:
        data = request.get_json()
        
        if not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Email and password required'}), 400
        
        user = User.query.filter_by(email=data['email'].lower().strip()).first()
        
        if not user or not user.check_password(data['password']):
            return jsonify({'error': 'Invalid email or password'}), 401
        
        if not user.is_active:
            return jsonify({'error': 'Account is deactivated'}), 401
        
        # Generate access token
        access_token = create_access_token(identity=user.id)
        
        logger.info(f"User logged in: {user.email}")
        
        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict(include_sensitive=True),
            'access_token': access_token
        })
        
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get user profile (requires authentication)"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get_or_404(user_id)
        
        return jsonify({
            'user': user.to_dict(),
            'addresses': [addr.to_dict() for addr in user.addresses]
        })
        
    except Exception as e:
        logger.error(f"Error fetching profile: {str(e)}")
        return jsonify({'error': 'Failed to fetch profile'}), 500

@app.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update user profile"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get_or_404(user_id)
        data = request.get_json()
        
        # Update allowed fields
        if 'first_name' in data:
            user.first_name = data['first_name'].strip()
        if 'last_name' in data:
            user.last_name = data['last_name'].strip()
        if 'phone' in data:
            user.phone = data['phone'].strip()
        
        user.updated_at = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"Profile updated for user: {user.email}")
        return jsonify(user.to_dict())
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating profile: {str(e)}")
        return jsonify({'error': 'Failed to update profile'}), 500

@app.route('/users/<int:user_id>/verify', methods=['GET'])
def verify_user(user_id):
    """Verify user exists and is active (for internal service calls)"""
    try:
        user = User.query.get_or_404(user_id)
        return jsonify({
            'user_id': user.id,
            'email': user.email,
            'is_active': user.is_active,
            'is_admin': user.is_admin
        })
    except Exception as e:
        logger.error(f"Error verifying user {user_id}: {str(e)}")
        return jsonify({'error': 'User not found'}), 404

# Address management
@app.route('/addresses', methods=['POST'])
@jwt_required()
def add_address():
    """Add a new address for the user"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        required_fields = ['street_address', 'city', 'state', 'postal_code', 'country']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Missing or empty field: {field}'}), 400
        
        # If this is the first address or marked as default, make it default
        is_default = data.get('is_default', False)
        if is_default or not Address.query.filter_by(user_id=user_id).first():
            # Unset other default addresses
            Address.query.filter_by(user_id=user_id, is_default=True).update({'is_default': False})
            is_default = True
        
        address = Address(
            user_id=user_id,
            street_address=data['street_address'].strip(),
            city=data['city'].strip(),
            state=data['state'].strip(),
            postal_code=data['postal_code'].strip(),
            country=data['country'].strip(),
            is_default=is_default
        )
        
        db.session.add(address)
        db.session.commit()
        
        return jsonify(address.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error adding address: {str(e)}")
        return jsonify({'error': 'Failed to add address'}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5002)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )
```

**User Service Requirements:**
```python
# user_service/requirements.txt
Flask==2.3.3
Flask-SQLAlchemy==3.0.5
Flask-Bcrypt==1.0.1
Flask-JWT-Extended==4.5.2
psycopg2-binary==2.9.7
python-dotenv==1.0.0
gunicorn==21.2.0
```

#### 3. Shopping Cart Service
**Responsibilities**: Cart management, session handling, cart persistence

**Complete Cart Service Implementation:**
```python
# cart_service/app.py
from flask import Flask, request, jsonify
import redis
import json
import os
import logging
from datetime import datetime, timedelta
import requests

app = Flask(__name__)

# Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')

PRODUCT_SERVICE_URL = os.getenv('PRODUCT_SERVICE_URL', 'http://localhost:5001')
CART_EXPIRY_HOURS = int(os.getenv('CART_EXPIRY_HOURS', 24))

# Redis connection
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    decode_responses=True
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CartService:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.expiry_seconds = CART_EXPIRY_HOURS * 3600
    
    def get_cart_key(self, user_id):
        return f"cart:{user_id}"
    
    def get_cart(self, user_id):
        """Get user's cart from Redis"""
        try:
            cart_data = self.redis.get(self.get_cart_key(user_id))
            if cart_data:
                cart = json.loads(cart_data)
                # Refresh expiry
                self.redis.expire(self.get_cart_key(user_id), self.expiry_seconds)
                return cart
            return {'items': [], 'created_at': datetime.utcnow().isoformat()}
        except Exception as e:
            logger.error(f"Error fetching cart for user {user_id}: {str(e)}")
            return {'items': [], 'created_at': datetime.utcnow().isoformat()}
    
    def save_cart(self, user_id, cart):
        """Save cart to Redis with expiry"""
        try:
            cart['updated_at'] = datetime.utcnow().isoformat()
            self.redis.setex(
                self.get_cart_key(user_id),
                self.expiry_seconds,
                json.dumps(cart)
            )
            return True
        except Exception as e:
            logger.error(f"Error saving cart for user {user_id}: {str(e)}")
            return False
    
    def add_item(self, user_id, product_id, quantity):
        """Add item to cart"""
        cart = self.get_cart(user_id)
        
        # Find existing item
        existing_item = None
        for item in cart['items']:
            if item['product_id'] == product_id:
                existing_item = item
                break
        
        if existing_item:
            existing_item['quantity'] += quantity
        else:
            cart['items'].append({
                'product_id': product_id,
                'quantity': quantity,
                'added_at': datetime.utcnow().isoformat()
            })
        
        return self.save_cart(user_id, cart)
    
    def remove_item(self, user_id, product_id):
        """Remove item from cart"""
        cart = self.get_cart(user_id)
        cart['items'] = [item for item in cart['items'] if item['product_id'] != product_id]
        return self.save_cart(user_id, cart)
    
    def update_quantity(self, user_id, product_id, quantity):
        """Update item quantity in cart"""
        if quantity <= 0:
            return self.remove_item(user_id, product_id)
        
        cart = self.get_cart(user_id)
        
        for item in cart['items']:
            if item['product_id'] == product_id:
                item['quantity'] = quantity
                item['updated_at'] = datetime.utcnow().isoformat()
                break
        
        return self.save_cart(user_id, cart)
    
    def clear_cart(self, user_id):
        """Clear all items from cart"""
        try:
            self.redis.delete(self.get_cart_key(user_id))
            return True
        except Exception as e:
            logger.error(f"Error clearing cart for user {user_id}: {str(e)}")
            return False

cart_service = CartService(redis_client)

def get_product_details(product_ids):
    """Fetch product details from Product Service"""
    try:
        if not product_ids:
            return {}
        
        # Make batch request to product service
        response = requests.get(
            f"{PRODUCT_SERVICE_URL}/products",
            params={'ids': ','.join(map(str, product_ids))},
            timeout=5
        )
        
        if response.status_code == 200:
            products_data = response.json()
            products = products_data.get('products', [])
            return {product['id']: product for product in products}
        
        logger.warning(f"Product service returned {response.status_code}")
        return {}
        
    except requests.RequestException as e:
        logger.error(f"Error fetching product details: {str(e)}")
        return {}

def enrich_cart_with_product_details(cart):
    """Add product details to cart items"""
    if not cart['items']:
        return cart
    
    product_ids = [item['product_id'] for item in cart['items']]
    products = get_product_details(product_ids)
    
    enriched_items = []
    total_amount = 0
    
    for item in cart['items']:
        product = products.get(item['product_id'])
        if product:
            enriched_item = {
                **item,
                'product_name': product['name'],
                'product_price': product['price'],
                'subtotal': product['price'] * item['quantity'],
                'stock_available': product['stock_quantity']
            }
            total_amount += enriched_item['subtotal']
            enriched_items.append(enriched_item)
        else:
            # Product not found, mark as unavailable
            enriched_items.append({
                **item,
                'product_name': 'Product Unavailable',
                'product_price': 0,
                'subtotal': 0,
                'stock_available': 0,
                'unavailable': True
            })
    
    cart['items'] = enriched_items
    cart['total_amount'] = total_amount
    cart['total_items'] = sum(item['quantity'] for item in enriched_items if not item.get('unavailable'))
    
    return cart

# Routes
@app.route('/health', methods=['GET'])
def health_check():
    try:
        redis_client.ping()
        return jsonify({'status': 'healthy', 'service': 'cart-service', 'redis': 'connected'})
    except:
        return jsonify({'status': 'unhealthy', 'service': 'cart-service', 'redis': 'disconnected'}), 503

@app.route('/cart/<int:user_id>', methods=['GET'])
def get_cart(user_id):
    """Get user's cart with product details"""
    try:
        cart = cart_service.get_cart(user_id)
        enriched_cart = enrich_cart_with_product_details(cart)
        return jsonify(enriched_cart)
    except Exception as e:
        logger.error(f"Error fetching cart for user {user_id}: {str(e)}")
        return jsonify({'error': 'Failed to fetch cart'}), 500

@app.route('/cart/<int:user_id>/items', methods=['POST'])
def add_to_cart(user_id):
    """Add item to cart"""
    try:
        data = request.get_json()
        
        if not data.get('product_id') or not data.get('quantity'):
            return jsonify({'error': 'product_id and quantity are required'}), 400
        
        product_id = int(data['product_id'])
        quantity = int(data['quantity'])
        
        if quantity <= 0:
            return jsonify({'error': 'Quantity must be positive'}), 400
        
        # Verify product exists and has enough stock
        products = get_product_details([product_id])
        product = products.get(product_id)
        
        if not product:
            return jsonify({'error': 'Product not found'}), 404
        
        if product['stock_quantity'] < quantity:
            return jsonify({'error': 'Insufficient stock'}), 400
        
        success = cart_service.add_item(user_id, product_id, quantity)
        
        if success:
            cart = cart_service.get_cart(user_id)
            enriched_cart = enrich_cart_with_product_details(cart)
            return jsonify(enriched_cart), 201
        else:
            return jsonify({'error': 'Failed to add item to cart'}), 500
            
    except ValueError:
        return jsonify({'error': 'Invalid product_id or quantity'}), 400
    except Exception as e:
        logger.error(f"Error adding item to cart for user {user_id}: {str(e)}")
        return jsonify({'error': 'Failed to add item to cart'}), 500

@app.route('/cart/<int:user_id>/items/<int:product_id>', methods=['PUT'])
def update_cart_item(user_id, product_id):
    """Update item quantity in cart"""
    try:
        data = request.get_json()
        
        if 'quantity' not in data:
            return jsonify({'error': 'quantity is required'}), 400
        
        quantity = int(data['quantity'])
        
        if quantity < 0:
            return jsonify({'error': 'Quantity cannot be negative'}), 400
        
        # Verify product exists and has enough stock
        if quantity > 0:
            products = get_product_details([product_id])
            product = products.get(product_id)
            
            if not product:
                return jsonify({'error': 'Product not found'}), 404
            
            if product['stock_quantity'] < quantity:
                return jsonify({'error': 'Insufficient stock'}), 400
        
        success = cart_service.update_quantity(user_id, product_id, quantity)
        
        if success:
            cart = cart_service.get_cart(user_id)
            enriched_cart = enrich_cart_with_product_details(cart)
            return jsonify(enriched_cart)
        else:
            return jsonify({'error': 'Failed to update cart item'}), 500
            
    except ValueError:
        return jsonify({'error': 'Invalid quantity'}), 400
    except Exception as e:
        logger.error(f"Error updating cart item for user {user_id}: {str(e)}")
        return jsonify({'error': 'Failed to update cart item'}), 500

@app.route('/cart/<int:user_id>/items/<int:product_id>', methods=['DELETE'])
def remove_from_cart(user_id, product_id):
    """Remove item from cart"""
    try:
        success = cart_service.remove_item(user_id, product_id)
        
        if success:
            cart = cart_service.get_cart(user_id)
            enriched_cart = enrich_cart_with_product_details(cart)
            return jsonify(enriched_cart)
        else:
            return jsonify({'error': 'Failed to remove item from cart'}), 500
            
    except Exception as e:
        logger.error(f"Error removing item from cart for user {user_id}: {str(e)}")
        return jsonify({'error': 'Failed to remove item from cart'}), 500

@app.route('/cart/<int:user_id>', methods=['DELETE'])
def clear_cart(user_id):
    """Clear entire cart"""
    try:
        success = cart_service.clear_cart(user_id)
        
        if success:
            return jsonify({'message': 'Cart cleared successfully'})
        else:
            return jsonify({'error': 'Failed to clear cart'}), 500
            
    except Exception as e:
        logger.error(f"Error clearing cart for user {user_id}: {str(e)}")
        return jsonify({'error': 'Failed to clear cart'}), 500

@app.route('/cart/<int:user_id>/validate', methods=['POST'])
def validate_cart(user_id):
    """Validate cart items against current product availability and prices"""
    try:
        cart = cart_service.get_cart(user_id)
        enriched_cart = enrich_cart_with_product_details(cart)
        
        validation_issues = []
        
        for item in enriched_cart['items']:
            if item.get('unavailable'):
                validation_issues.append({
                    'product_id': item['product_id'],
                    'issue': 'Product no longer available'
                })
            elif item['stock_available'] < item['quantity']:
                validation_issues.append({
                    'product_id': item['product_id'],
                    'issue': f'Only {item["stock_available"]} items available, requested {item["quantity"]}'
                })
        
        return jsonify({
            'cart': enriched_cart,
            'is_valid': len(validation_issues) == 0,
            'validation_issues': validation_issues
        })
        
    except Exception as e:
        logger.error(f"Error validating cart for user {user_id}: {str(e)}")
        return jsonify({'error': 'Failed to validate cart'}), 500

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5003)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )
```

**Cart Service Requirements:**
```python
# cart_service/requirements.txt
Flask==2.3.3
redis==4.6.0
requests==2.31.0
python-dotenv==1.0.0
gunicorn==21.2.0
```

### Phase 2: Docker Configuration and Service Orchestration

#### Docker Compose Setup
**Complete multi-service orchestration configuration:**

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Databases
  postgres-products:
    image: postgres:15
    environment:
      POSTGRES_DB: products_db
      POSTGRES_USER: productuser
      POSTGRES_PASSWORD: productpass
    volumes:
      - postgres_products_data:/var/lib/postgresql/data
      - ./product_service/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - ecommerce-network

  postgres-users:
    image: postgres:15
    environment:
      POSTGRES_DB: users_db
      POSTGRES_USER: useruser
      POSTGRES_PASSWORD: userpass
    volumes:
      - postgres_users_data:/var/lib/postgresql/data
      - ./user_service/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5433:5432"
    networks:
      - ecommerce-network

  postgres-orders:
    image: postgres:15
    environment:
      POSTGRES_DB: orders_db
      POSTGRES_USER: orderuser
      POSTGRES_PASSWORD: orderpass
    volumes:
      - postgres_orders_data:/var/lib/postgresql/data
      - ./order_service/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5434:5432"
    networks:
      - ecommerce-network

  redis-cart:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_cart_data:/data
    ports:
      - "6379:6379"
    networks:
      - ecommerce-network

  mongodb-notifications:
    image: mongo:6
    environment:
      MONGO_INITDB_ROOT_USERNAME: notifuser
      MONGO_INITDB_ROOT_PASSWORD: notifpass
    volumes:
      - mongodb_notifications_data:/data/db
    ports:
      - "27017:27017"
    networks:
      - ecommerce-network

  # Message Queue
  rabbitmq:
    image: rabbitmq:3.12-management
    environment:
      RABBITMQ_DEFAULT_USER: rabbitmq
      RABBITMQ_DEFAULT_PASS: rabbitmq
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    ports:
      - "5672:5672"
      - "15672:15672"  # Management UI
    networks:
      - ecommerce-network

  # Microservices
  product-service:
    build:
      context: ./product_service
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: postgresql://productuser:productpass@postgres-products:5432/products_db
      FLASK_ENV: production
      PORT: 5001
    depends_on:
      - postgres-products
    ports:
      - "5001:5001"
    networks:
      - ecommerce-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  user-service:
    build:
      context: ./user_service
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: postgresql://useruser:userpass@postgres-users:5432/users_db
      JWT_SECRET_KEY: your-super-secret-jwt-key-change-in-production
      FLASK_ENV: production
      PORT: 5002
    depends_on:
      - postgres-users
    ports:
      - "5002:5002"
    networks:
      - ecommerce-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5002/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  cart-service:
    build:
      context: ./cart_service
      dockerfile: Dockerfile
    environment:
      REDIS_HOST: redis-cart
      REDIS_PORT: 6379
      PRODUCT_SERVICE_URL: http://product-service:5001
      FLASK_ENV: production
      PORT: 5003
    depends_on:
      - redis-cart
      - product-service
    ports:
      - "5003:5003"
    networks:
      - ecommerce-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5003/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  order-service:
    build:
      context: ./order_service
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: postgresql://orderuser:orderpass@postgres-orders:5432/orders_db
      PRODUCT_SERVICE_URL: http://product-service:5001
      CART_SERVICE_URL: http://cart-service:5003
      PAYMENT_SERVICE_URL: http://payment-service:5005
      USER_SERVICE_URL: http://user-service:5002
      RABBITMQ_URL: amqp://rabbitmq:rabbitmq@rabbitmq:5672/
      FLASK_ENV: production
      PORT: 5004
    depends_on:
      - postgres-orders
      - rabbitmq
      - product-service
      - cart-service
      - user-service
    ports:
      - "5004:5004"
    networks:
      - ecommerce-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5004/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  payment-service:
    build:
      context: ./payment_service
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: postgresql://orderuser:orderpass@postgres-orders:5432/orders_db
      STRIPE_SECRET_KEY: sk_test_your_stripe_secret_key
      RABBITMQ_URL: amqp://rabbitmq:rabbitmq@rabbitmq:5672/
      FLASK_ENV: production
      PORT: 5005
    depends_on:
      - postgres-orders
      - rabbitmq
    ports:
      - "5005:5005"
    networks:
      - ecommerce-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5005/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  notification-service:
    build:
      context: ./notification_service
      dockerfile: Dockerfile
    environment:
      MONGODB_URL: mongodb://notifuser:notifpass@mongodb-notifications:27017/
      RABBITMQ_URL: amqp://rabbitmq:rabbitmq@rabbitmq:5672/
      SMTP_HOST: smtp.gmail.com
      SMTP_PORT: 587
      SMTP_USERNAME: your-email@gmail.com
      SMTP_PASSWORD: your-app-password
      FLASK_ENV: production
      PORT: 5006
    depends_on:
      - mongodb-notifications
      - rabbitmq
    ports:
      - "5006:5006"
    networks:
      - ecommerce-network

  # API Gateway
  api-gateway:
    build:
      context: ./api_gateway
      dockerfile: Dockerfile
    environment:
      JWT_SECRET_KEY: your-super-secret-jwt-key-change-in-production
      USER_SERVICE_URL: http://user-service:5002
      PRODUCT_SERVICE_URL: http://product-service:5001
      CART_SERVICE_URL: http://cart-service:5003
      ORDER_SERVICE_URL: http://order-service:5004
      PAYMENT_SERVICE_URL: http://payment-service:5005
      FLASK_ENV: production
      PORT: 8000
    depends_on:
      - user-service
      - product-service
      - cart-service
      - order-service
      - payment-service
    ports:
      - "8000:8000"
    networks:
      - ecommerce-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Monitoring and Observability
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    ports:
      - "9090:9090"
    networks:
      - ecommerce-network

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    ports:
      - "3000:3000"
    networks:
      - ecommerce-network

  jaeger:
    image: jaegertracing/all-in-one:latest
    environment:
      COLLECTOR_ZIPKIN_HOST_PORT: :9411
    ports:
      - "16686:16686"  # Jaeger UI
      - "14268:14268"  # Collector HTTP
    networks:
      - ecommerce-network

volumes:
  postgres_products_data:
  postgres_users_data:
  postgres_orders_data:
  redis_cart_data:
  mongodb_notifications_data:
  rabbitmq_data:
  prometheus_data:
  grafana_data:

networks:
  ecommerce-network:
    driver: bridge
```

#### Environment Configuration
```bash
# .env file for development
# Database URLs
POSTGRES_PRODUCTS_URL=postgresql://productuser:productpass@localhost:5432/products_db
POSTGRES_USERS_URL=postgresql://useruser:userpass@localhost:5433/users_db
POSTGRES_ORDERS_URL=postgresql://orderuser:orderpass@localhost:5434/orders_db

# Redis
REDIS_URL=redis://localhost:6379/0

# MongoDB
MONGODB_URL=mongodb://notifuser:notifpass@localhost:27017/notifications

# RabbitMQ
RABBITMQ_URL=amqp://rabbitmq:rabbitmq@localhost:5672/

# JWT Secret (change in production)
JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production

# External APIs
STRIPE_SECRET_KEY=sk_test_your_stripe_secret_key
STRIPE_PUBLISHABLE_KEY=pk_test_your_stripe_publishable_key

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Service URLs (for local development)
USER_SERVICE_URL=http://localhost:5002
PRODUCT_SERVICE_URL=http://localhost:5001
CART_SERVICE_URL=http://localhost:5003
ORDER_SERVICE_URL=http://localhost:5004
PAYMENT_SERVICE_URL=http://localhost:5005
NOTIFICATION_SERVICE_URL=http://localhost:5006
```

#### Deployment Scripts
```bash
#!/bin/bash
# scripts/deploy.sh

echo "🚀 Deploying E-Commerce Microservices..."

# Build all services
echo "📦 Building services..."
docker-compose build

# Start databases first
echo "🗄️ Starting databases..."
docker-compose up -d postgres-products postgres-users postgres-orders redis-cart mongodb-notifications rabbitmq

# Wait for databases to be ready
echo "⏳ Waiting for databases to be ready..."
sleep 30

# Run database migrations
echo "🔄 Running database migrations..."
docker-compose run --rm product-service flask db upgrade
docker-compose run --rm user-service flask db upgrade
docker-compose run --rm order-service flask db upgrade

# Start all services
echo "🌟 Starting all services..."
docker-compose up -d

echo "✅ Deployment complete!"
echo "🌐 API Gateway: http://localhost:8000"
echo "📊 Grafana: http://localhost:3000 (admin/admin)"
echo "🔍 Jaeger: http://localhost:16686"
echo "📈 Prometheus: http://localhost:9090"
echo "🐰 RabbitMQ Management: http://localhost:15672 (rabbitmq/rabbitmq)"
```

```bash
#!/bin/bash
# scripts/test.sh

echo "🧪 Running integration tests..."

# Wait for services to be ready
echo "⏳ Waiting for services..."
sleep 60

# Run health checks
echo "🏥 Checking service health..."
services=("api-gateway:8000" "product-service:5001" "user-service:5002" "cart-service:5003" "order-service:5004" "payment-service:5005")

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if curl -f "http://localhost:$port/health" > /dev/null 2>&1; then
        echo "✅ $name is healthy"
    else
        echo "❌ $name is not responding"
        exit 1
    fi
done

# Run API tests
echo "🔗 Running API tests..."
python tests/integration_tests.py

echo "✅ All tests passed!"
```

### Phase 3: Testing and Quality Assurance

#### Integration Testing Suite
```python
# tests/integration_tests.py
import requests
import json
import time
import pytest
from datetime import datetime

BASE_URL = "http://localhost:8000/api"

class TestECommerceAPI:
    def __init__(self):
        self.session = requests.Session()
        self.user_token = None
        self.user_id = None
        self.product_id = None
        self.order_id = None
    
    def test_user_registration_and_authentication(self):
        """Test user registration and login flow"""
        print("🧪 Testing user registration and authentication...")
        
        # Test user registration
        user_data = {
            "email": f"test_{int(time.time())}@example.com",
            "password": "TestPassword123",
            "first_name": "Test",
            "last_name": "User"
        }
        
        response = self.session.post(f"{BASE_URL}/auth/register", json=user_data)
        assert response.status_code == 201, f"Registration failed: {response.text}"
        
        reg_data = response.json()
        self.user_token = reg_data["access_token"]
        self.user_id = reg_data["user"]["id"]
        
        # Set authorization header for subsequent requests
        self.session.headers.update({"Authorization": f"Bearer {self.user_token}"})
        
        # Test login
        login_data = {
            "email": user_data["email"],
            "password": user_data["password"]
        }
        
        response = self.session.post(f"{BASE_URL}/auth/login", json=login_data)
        assert response.status_code == 200, f"Login failed: {response.text}"
        
        print("✅ User registration and authentication passed")
    
    def test_product_catalog(self):
        """Test product catalog functionality"""
        print("🧪 Testing product catalog...")
        
        # Get all products
        response = self.session.get(f"{BASE_URL}/products")
        assert response.status_code == 200, f"Get products failed: {response.text}"
        
        products_data = response.json()
        assert "products" in products_data
        
        if products_data["products"]:
            self.product_id = products_data["products"][0]["id"]
            
            # Get specific product
            response = self.session.get(f"{BASE_URL}/products/{self.product_id}")
            assert response.status_code == 200, f"Get specific product failed: {response.text}"
        
        # Get categories
        response = self.session.get(f"{BASE_URL}/categories")
        assert response.status_code == 200, f"Get categories failed: {response.text}"
        
        print("✅ Product catalog tests passed")
    
    def test_shopping_cart(self):
        """Test shopping cart functionality"""
        print("🧪 Testing shopping cart...")
        
        if not self.product_id:
            print("⚠️ Skipping cart tests - no products available")
            return
        
        # Get empty cart
        response = self.session.get(f"{BASE_URL}/cart")
        assert response.status_code == 200, f"Get cart failed: {response.text}"
        
        cart_data = response.json()
        assert "items" in cart_data
        
        # Add item to cart
        add_item_data = {
            "product_id": self.product_id,
            "quantity": 2
        }
        
        response = self.session.post(f"{BASE_URL}/cart/items", json=add_item_data)
        assert response.status_code == 201, f"Add to cart failed: {response.text}"
        
        cart_data = response.json()
        assert len(cart_data["items"]) > 0
        assert cart_data["items"][0]["product_id"] == self.product_id
        assert cart_data["items"][0]["quantity"] == 2
        
        # Update item quantity
        update_data = {"quantity": 3}
        response = self.session.put(f"{BASE_URL}/cart/items/{self.product_id}", json=update_data)
        assert response.status_code == 200, f"Update cart item failed: {response.text}"
        
        cart_data = response.json()
        assert cart_data["items"][0]["quantity"] == 3
        
        # Remove item from cart
        response = self.session.delete(f"{BASE_URL}/cart/items/{self.product_id}")
        assert response.status_code == 200, f"Remove from cart failed: {response.text}"
        
        print("✅ Shopping cart tests passed")
    
    def test_order_workflow(self):
        """Test complete order workflow"""
        print("🧪 Testing order workflow...")
        
        if not self.product_id:
            print("⚠️ Skipping order tests - no products available")
            return
        
        # Add item to cart for order
        add_item_data = {
            "product_id": self.product_id,
            "quantity": 1
        }
        
        response = self.session.post(f"{BASE_URL}/cart/items", json=add_item_data)
        assert response.status_code == 201
        
        # Create order
        order_data = {
            "shipping_address": {
                "street_address": "123 Test St",
                "city": "Test City",
                "state": "Test State",
                "postal_code": "12345",
                "country": "Test Country"
            }
        }
        
        response = self.session.post(f"{BASE_URL}/orders", json=order_data)
        assert response.status_code == 201, f"Create order failed: {response.text}"
        
        order_response = response.json()
        self.order_id = order_response["order"]["id"]
        
        # Get orders list
        response = self.session.get(f"{BASE_URL}/orders")
        assert response.status_code == 200, f"Get orders failed: {response.text}"
        
        orders_data = response.json()
        assert len(orders_data["orders"]) > 0
        
        # Get specific order
        response = self.session.get(f"{BASE_URL}/orders/{self.order_id}")
        assert response.status_code == 200, f"Get specific order failed: {response.text}"
        
        print("✅ Order workflow tests passed")
    
    def test_rate_limiting(self):
        """Test API rate limiting"""
        print("🧪 Testing rate limiting...")
        
        # Make multiple rapid requests to trigger rate limiting
        rate_limit_hit = False
        for i in range(60):  # Try to exceed rate limit
            response = self.session.get(f"{BASE_URL}/products")
            if response.status_code == 429:
                rate_limit_hit = True
                break
            time.sleep(0.1)
        
        if rate_limit_hit:
            print("✅ Rate limiting is working")
        else:
            print("⚠️ Rate limiting may not be properly configured")
    
    def run_all_tests(self):
        """Run all integration tests"""
        try:
            self.test_user_registration_and_authentication()
            self.test_product_catalog()
            self.test_shopping_cart()
            self.test_order_workflow()
            self.test_rate_limiting()
            print("\n🎉 All integration tests passed!")
            return True
        except AssertionError as e:
            print(f"\n❌ Test failed: {e}")
            return False
        except Exception as e:
            print(f"\n💥 Unexpected error: {e}")
            return False

if __name__ == "__main__":
    # Wait for services to be ready
    print("⏳ Waiting for services to start...")
    time.sleep(10)
    
    # Check if API Gateway is accessible
    try:
        response = requests.get(f"{BASE_URL}/../health", timeout=5)
        if response.status_code != 200:
            print("❌ API Gateway is not accessible")
            exit(1)
    except requests.RequestException:
        print("❌ Cannot connect to API Gateway")
        exit(1)
    
    # Run tests
    test_suite = TestECommerceAPI()
    success = test_suite.run_all_tests()
    
    exit(0 if success else 1)
```

#### Load Testing with Locust
```python
# tests/load_test.py
from locust import HttpUser, task, between
import random
import json

class ECommerceUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup user session"""
        # Register a test user
        user_data = {
            "email": f"loadtest_{random.randint(1000, 9999)}@example.com",
            "password": "TestPassword123",
            "first_name": "Load",
            "last_name": "Test"
        }
        
        response = self.client.post("/api/auth/register", json=user_data)
        if response.status_code == 201:
            data = response.json()
            self.token = data["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.token = None
            self.headers = {}
    
    @task(3)
    def browse_products(self):
        """Browse product catalog"""
        self.client.get("/api/products")
        
        # Sometimes browse categories
        if random.random() < 0.3:
            self.client.get("/api/categories")
    
    @task(2)
    def view_product_details(self):
        """View specific product"""
        # Get products list first
        response = self.client.get("/api/products")
        if response.status_code == 200:
            products = response.json().get("products", [])
            if products:
                product_id = random.choice(products)["id"]
                self.client.get(f"/api/products/{product_id}")
    
    @task(1)
    def manage_cart(self):
        """Add/remove items from cart"""
        if not self.token:
            return
        
        # Get current cart
        response = self.client.get("/api/cart", headers=self.headers)
        
        # Get products to add
        response = self.client.get("/api/products")
        if response.status_code == 200:
            products = response.json().get("products", [])
            if products:
                product = random.choice(products)
                
                # Add to cart
                add_data = {
                    "product_id": product["id"],
                    "quantity": random.randint(1, 3)
                }
                self.client.post("/api/cart/items", json=add_data, headers=self.headers)
    
    @task(1)
    def view_profile(self):
        """View user profile"""
        if not self.token:
            return
        
        self.client.get("/api/users/profile", headers=self.headers)

# Run with: locust -f tests/load_test.py --host=http://localhost:8000
```

#### Contract Testing with Pact
```python
# tests/contract_tests.py
import unittest
from pact import Consumer, Provider, Like, Term
import requests

class ProductServiceContractTest(unittest.TestCase):
    def setUp(self):
        self.pact = Consumer('api-gateway').has_pact_with(Provider('product-service'))
        self.pact.start()
    
    def tearDown(self):
        self.pact.stop()
    
    def test_get_products(self):
        """Test product service contract for getting products"""
        (self.pact
         .given('products exist')
         .upon_receiving('a request for products')
         .with_request('GET', '/products')
         .will_respond_with(200, body={
             'products': Like([{
                 'id': Like(1),
                 'name': Like('Test Product'),
                 'price': Like(29.99),
                 'stock_quantity': Like(10)
             }]),
             'pagination': Like({
                 'page': Like(1),
                 'pages': Like(1),
                 'total': Like(1)
             })
         }))
        
        # Make the actual request
        response = requests.get(f"{self.pact.uri}/products")
        
        # Verify the contract
        self.assertEqual(response.status_code, 200)
        self.pact.verify()

if __name__ == '__main__':
    unittest.main()
```

## Learning Objectives and Outcomes

### Technical Skills Developed

#### Microservices Architecture Patterns
By completing this project, you will master:

✅ **Service Decomposition**: Learn how to break down a monolithic application into loosely coupled services
✅ **Database per Service**: Implement polyglot persistence with different databases for different services
✅ **API Gateway Pattern**: Centralize cross-cutting concerns like authentication, rate limiting, and routing
✅ **Service Discovery**: Understand how services find and communicate with each other
✅ **Circuit Breaker Pattern**: Implement fault tolerance and graceful degradation
✅ **Event-Driven Architecture**: Use message queues for asynchronous communication
✅ **CQRS (Command Query Responsibility Segregation)**: Separate read and write operations

#### Distributed Systems Concepts
✅ **Eventual Consistency**: Handle data consistency across distributed services
✅ **Distributed Transactions**: Implement saga patterns for cross-service transactions
✅ **Load Balancing**: Distribute traffic across multiple service instances
✅ **Fault Tolerance**: Design systems that continue working despite failures
✅ **Scalability Patterns**: Scale services independently based on demand
✅ **Security**: Implement JWT authentication and authorization

#### DevOps and Deployment
✅ **Containerization**: Package services using Docker
✅ **Orchestration**: Use Docker Compose for multi-service deployment
✅ **Monitoring**: Implement logging, metrics, and distributed tracing
✅ **CI/CD**: Set up automated testing and deployment pipelines
✅ **Infrastructure as Code**: Define infrastructure using configuration files

### Assessment Criteria

#### Phase 1: Core Services (40 points)
- **Product Service** (10 points): Complete CRUD operations, inventory management
- **User Service** (10 points): Authentication, JWT tokens, profile management
- **Cart Service** (10 points): Redis-based session management, cart persistence
- **Order Service** (10 points): Order processing, workflow management

#### Phase 2: Integration & Communication (30 points)
- **API Gateway** (10 points): Request routing, authentication, rate limiting
- **Inter-Service Communication** (10 points): HTTP calls, error handling
- **Message Queues** (10 points): Asynchronous event processing

#### Phase 3: DevOps & Quality (30 points)
- **Containerization** (10 points): Docker images, Docker Compose
- **Testing** (10 points): Unit, integration, load testing
- **Monitoring** (10 points): Logging, metrics, health checks

### Bonus Challenges (Extra Credit)

#### Advanced Patterns (+20 points)
1. **Implement Event Sourcing**: Store all changes as events
2. **Add Caching Layer**: Implement Redis caching for read operations
3. **Implement SAGA Pattern**: Handle distributed transactions
4. **Add Search Service**: Elasticsearch for product search
5. **Implement Service Mesh**: Use Istio for advanced service communication

#### Performance Optimization (+15 points)
1. **Database Optimization**: Query optimization, indexing strategies
2. **Caching Strategy**: Multi-level caching (service, gateway, CDN)
3. **Load Testing**: Achieve 1000+ concurrent users
4. **Auto-scaling**: Implement horizontal pod autoscaling

#### Security Enhancements (+15 points)
1. **OAuth2 Integration**: Social login (Google, Facebook)
2. **API Security**: Input validation, SQL injection prevention
3. **Encryption**: Data encryption at rest and in transit
4. **Audit Logging**: Track all user actions

### Self-Assessment Checklist

#### Week 1: Foundation
□ Set up development environment with Docker  
□ Implement Product Service with PostgreSQL  
□ Implement User Service with JWT authentication  
□ Create basic API Gateway with routing  
□ Write unit tests for core services  

#### Week 2: Integration
□ Implement Cart Service with Redis  
□ Add Order Service with workflow management  
□ Implement inter-service communication  
□ Add message queue for asynchronous processing  
□ Set up monitoring and logging  

#### Week 3: Quality & Deployment
□ Write comprehensive integration tests  
□ Implement load testing with realistic scenarios  
□ Set up CI/CD pipeline  
□ Deploy to cloud platform (AWS/GCP/Azure)  
□ Document API endpoints and architecture  

#### Week 4: Advanced Features
□ Implement advanced patterns (Event Sourcing, CQRS)  
□ Add performance monitoring and optimization  
□ Implement security best practices  
□ Create user documentation and deployment guide  
□ Present project and demonstrate scalability  

### Common Pitfalls to Avoid

#### Design Issues
❌ **Creating too fine-grained services**: Start with logical boundaries
❌ **Shared databases**: Each service should own its data
❌ **Synchronous everything**: Use async communication where appropriate
❌ **Ignoring failure scenarios**: Design for failure from the start

#### Implementation Issues
❌ **No error handling**: Implement proper error handling and retries
❌ **Missing authentication**: Secure all service endpoints
❌ **No monitoring**: Add health checks and logging from day one
❌ **Testing only happy paths**: Test failure scenarios and edge cases

#### Deployment Issues
❌ **No service discovery**: Services need to find each other reliably
❌ **Missing environment configs**: Use environment variables for configuration
❌ **No resource limits**: Set memory and CPU limits for containers
❌ **Ignoring dependencies**: Start services in correct order

### Real-World Applications

This project simulates real enterprise scenarios found in:

🏢 **E-commerce Platforms**: Amazon, eBay, Shopify  
🏦 **Financial Services**: Payment processing, banking systems  
🎯 **SaaS Applications**: Multi-tenant software platforms  
🎮 **Gaming Platforms**: User management, in-game purchases  
📱 **Mobile Applications**: Backend services for mobile apps  

### Career Relevance

Skills developed in this project are highly valued for:

👨‍💻 **Software Engineer**: Microservices development  
🏗️ **Solutions Architect**: System design and architecture  
⚙️ **DevOps Engineer**: Container orchestration and deployment  
☁️ **Cloud Engineer**: Cloud-native application development  
🔧 **Site Reliability Engineer**: System monitoring and scaling  

### Next Steps

After completing this project, consider:

1. **Deploy to Kubernetes**: Learn container orchestration
2. **Implement Service Mesh**: Add Istio for advanced networking
3. **Add Machine Learning**: Recommendation engine, fraud detection
4. **Mobile App**: Build React Native or Flutter frontend
5. **Advanced Analytics**: Real-time data processing with Apache Kafka

## Resources and References

### Documentation
- **Microservices Patterns**: https://microservices.io/patterns/
- **Docker Documentation**: https://docs.docker.com/
- **Flask Documentation**: https://flask.palletsprojects.com/
- **PostgreSQL Documentation**: https://www.postgresql.org/docs/
- **Redis Documentation**: https://redis.io/documentation

### Books
- "Microservices Patterns" by Chris Richardson
- "Building Microservices" by Sam Newman
- "Designing Data-Intensive Applications" by Martin Kleppmann
- "Site Reliability Engineering" by Google

### Tools and Platforms
- **Container Registry**: Docker Hub, AWS ECR, Google Container Registry
- **Cloud Platforms**: AWS, Google Cloud Platform, Microsoft Azure
- **Monitoring**: Prometheus, Grafana, Jaeger, ELK Stack
- **CI/CD**: GitHub Actions, GitLab CI, Jenkins, CircleCI

### Community
- **Reddit**: r/microservices, r/docker, r/flask
- **Stack Overflow**: Tags: microservices, docker, flask, postgresql
- **GitHub**: Explore microservices example repositories
- **Discord/Slack**: Join developer communities

---

*This project provides a comprehensive introduction to microservices architecture while building a real-world application. Take your time to understand each concept and don't hesitate to experiment with different approaches!*

# Project 3: Resilient Microservices Platform

*Duration: 3-4 weeks | Difficulty: Advanced | Prerequisites: Docker, Python/Node.js, Basic networking*

## Project Overview

Build a production-ready microservices platform that can gracefully handle failures, automatically recover from issues, and maintain service availability even when individual components fail. This project demonstrates enterprise-level resilience patterns used by companies like Netflix, Amazon, and Google.

### What You'll Build

1. **Multi-service Architecture**: 4-5 interconnected microservices
2. **Circuit Breaker Pattern**: Prevent cascading failures
3. **Retry Mechanisms**: Smart retry logic with exponential backoff
4. **Health Monitoring**: Comprehensive health checks and metrics
5. **Automated Recovery**: Self-healing capabilities
6. **Chaos Engineering**: Netflix-style fault injection testing
7. **Service Discovery**: Dynamic service registration and discovery
8. **Load Balancing**: Distribute traffic across service instances

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │────│  Load Balancer  │────│ Service Discovery│
│  (Kong/Nginx)   │    │   (HAProxy)     │    │   (Consul)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  User Service   │◄───┤ Circuit Breaker │───►│ Order Service   │
│   (Flask/FastAPI)│    │   (Hystrix)     │    │   (Express)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Payment Service │◄───┤ Retry Logic     │───►│ Notification    │
│   (Spring Boot) │    │ (Exponential)   │    │   Service       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Database     │    │   Monitoring    │    │  Chaos Monkey   │
│ (PostgreSQL)    │    │  (Prometheus)   │    │  (Fault Inject) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Business Context

**Problem Statement**: In a microservices architecture, services depend on each other. When one service fails, it can cause a cascade of failures that brings down the entire system. Traditional monolithic retry logic and error handling aren't sufficient for distributed systems.

**Real-World Scenario**: An e-commerce platform where:
- User Service manages authentication
- Order Service processes purchases  
- Payment Service handles transactions
- Notification Service sends confirmations

If the Payment Service becomes slow or unresponsive, it shouldn't crash the entire order flow.

## Example Code
```python
# Example: Health check endpoint
from flask import Flask
app = Flask(__name__)

@app.route('/health')
def health():
    return 'OK', 200

if __name__ == '__main__':
    app.run(port=5002)
```

## Core Implementation

### 1. Circuit Breaker Pattern

Circuit breakers prevent cascading failures by monitoring service calls and "opening" when failure rates exceed thresholds.

#### Python Circuit Breaker Implementation

```python
# circuit_breaker.py
import time
import logging
from enum import Enum
from typing import Callable, Any
from functools import wraps

class CircuitState(Enum):
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failing, rejecting calls
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered

class CircuitBreaker:
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: Exception = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
        self.logger = logging.getLogger(__name__)

    def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.logger.info("Circuit breaker entering HALF_OPEN state")
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
        return (time.time() - self.last_failure_time) >= self.recovery_timeout

    def _on_success(self):
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.logger.info("Circuit breaker reset to CLOSED state")

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")

# Decorator for easy use
def circuit_breaker(failure_threshold=5, recovery_timeout=60):
    def decorator(func):
        cb = CircuitBreaker(failure_threshold, recovery_timeout)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)
        return wrapper
    return decorator

# Usage example
@circuit_breaker(failure_threshold=3, recovery_timeout=30)
def call_payment_service(order_id: str, amount: float):
    """Call external payment service with circuit breaker protection"""
    import requests
    response = requests.post(
        "http://payment-service:8080/process",
        json={"order_id": order_id, "amount": amount},
        timeout=5
    )
    response.raise_for_status()
    return response.json()
```

#### Advanced Circuit Breaker with Metrics

```python
# advanced_circuit_breaker.py
import time
from collections import defaultdict
from threading import Lock
import prometheus_client

class MetricsCircuitBreaker:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.circuit_breaker = CircuitBreaker(**kwargs)
        self.lock = Lock()
        
        # Prometheus metrics
        self.request_counter = prometheus_client.Counter(
            f'circuit_breaker_requests_total',
            'Total requests through circuit breaker',
            ['name', 'status']
        )
        
        self.state_gauge = prometheus_client.Gauge(
            f'circuit_breaker_state',
            'Current circuit breaker state (0=CLOSED, 1=OPEN, 2=HALF_OPEN)',
            ['name']
        )
        
        self.failure_rate_gauge = prometheus_client.Gauge(
            f'circuit_breaker_failure_rate',
            'Current failure rate',
            ['name']
        )

    def call(self, func, *args, **kwargs):
        with self.lock:
            start_time = time.time()
            
            try:
                result = self.circuit_breaker.call(func, *args, **kwargs)
                self.request_counter.labels(name=self.name, status='success').inc()
                return result
                
            except Exception as e:
                self.request_counter.labels(name=self.name, status='failure').inc()
                raise e
                
            finally:
                # Update metrics
                state_mapping = {
                    CircuitState.CLOSED: 0,
                    CircuitState.OPEN: 1,
                    CircuitState.HALF_OPEN: 2
                }
                self.state_gauge.labels(name=self.name).set(
                    state_mapping[self.circuit_breaker.state]
                )
```

### 2. Retry Mechanisms with Exponential Backoff

Smart retry logic prevents overwhelming failing services while maximizing success chances.

```python
# retry_mechanism.py
import time
import random
import logging
from typing import Callable, Any, Optional
from functools import wraps

class RetryConfig:
    def __init__(self,
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

class RetryMechanism:
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.logger.info(f"Success on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    self.logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
        
        self.logger.error(f"All {self.config.max_attempts} attempts failed")
        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        # Exponential backoff: base_delay * (exponential_base ^ attempt)
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        
        # Cap at max_delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)

# Decorator for easy use
def retry(max_attempts=3, base_delay=1.0, max_delay=60.0, exponential_base=2.0, jitter=True):
    def decorator(func):
        config = RetryConfig(max_attempts, base_delay, max_delay, exponential_base, jitter)
        retry_mechanism = RetryMechanism(config)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return retry_mechanism.execute(func, *args, **kwargs)
        return wrapper
    return decorator

# Combined Circuit Breaker + Retry
class ResilientServiceClient:
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.circuit_breaker = MetricsCircuitBreaker(service_name)
        self.retry_config = RetryConfig(max_attempts=3, base_delay=0.5)
        self.retry_mechanism = RetryMechanism(self.retry_config)

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with both circuit breaker and retry protection"""
        def protected_call():
            return self.circuit_breaker.call(func, *args, **kwargs)
        
        return self.retry_mechanism.execute(protected_call)

# Example usage
payment_client = ResilientServiceClient("payment-service")

def process_payment(order_id: str, amount: float):
    return payment_client.call(call_payment_service, order_id, amount)
```

### 3. Health Check System

Comprehensive health monitoring for early failure detection and automated recovery.

```python
# health_check_system.py
import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import aiohttp
from prometheus_client import Gauge, Counter

class HealthStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"

@dataclass
class HealthCheckResult:
    service_name: str
    status: HealthStatus
    timestamp: datetime
    response_time_ms: float
    details: Dict = None
    error_message: Optional[str] = None

class HealthChecker:
    def __init__(self, service_name: str, check_url: str, timeout: float = 5.0):
        self.service_name = service_name
        self.check_url = check_url
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        # Metrics
        self.health_status_gauge = Gauge(
            'service_health_status',
            'Service health status (1=HEALTHY, 0.5=DEGRADED, 0=UNHEALTHY)',
            ['service_name']
        )
        
        self.response_time_gauge = Gauge(
            'health_check_response_time_seconds',
            'Health check response time',
            ['service_name']
        )
        
        self.health_check_counter = Counter(
            'health_checks_total',
            'Total health checks performed',
            ['service_name', 'status']
        )

    async def check_health(self) -> HealthCheckResult:
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.check_url, 
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        status = HealthStatus.HEALTHY
                        try:
                            details = await response.json()
                        except:
                            details = {"raw_response": await response.text()}
                    elif response.status in [500, 502, 503, 504]:
                        status = HealthStatus.UNHEALTHY
                        details = {"http_status": response.status}
                    else:
                        status = HealthStatus.DEGRADED
                        details = {"http_status": response.status}
                    
                    result = HealthCheckResult(
                        service_name=self.service_name,
                        status=status,
                        timestamp=datetime.now(),
                        response_time_ms=response_time,
                        details=details
                    )
                    
                    self._update_metrics(result)
                    return result
                    
        except asyncio.TimeoutError:
            result = HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error_message="Health check timeout"
            )
            self._update_metrics(result)
            return result
            
        except Exception as e:
            result = HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
            self._update_metrics(result)
            return result

    def _update_metrics(self, result: HealthCheckResult):
        # Update Prometheus metrics
        status_value = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.DEGRADED: 0.5,
            HealthStatus.UNHEALTHY: 0.0,
            HealthStatus.UNKNOWN: -1.0
        }[result.status]
        
        self.health_status_gauge.labels(service_name=self.service_name).set(status_value)
        self.response_time_gauge.labels(service_name=self.service_name).set(
            result.response_time_ms / 1000
        )
        self.health_check_counter.labels(
            service_name=self.service_name, 
            status=result.status.value
        ).inc()

class HealthMonitor:
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.health_checkers: List[HealthChecker] = []
        self.health_history: Dict[str, List[HealthCheckResult]] = {}
        self.alert_callbacks: List[Callable] = []
        self.logger = logging.getLogger(__name__)
        self.running = False

    def add_service(self, service_name: str, health_url: str, timeout: float = 5.0):
        """Add a service to monitor"""
        checker = HealthChecker(service_name, health_url, timeout)
        self.health_checkers.append(checker)
        self.health_history[service_name] = []

    def add_alert_callback(self, callback: Callable[[HealthCheckResult], None]):
        """Add callback for health alerts"""
        self.alert_callbacks.append(callback)

    async def start_monitoring(self):
        """Start continuous health monitoring"""
        self.running = True
        self.logger.info(f"Starting health monitoring for {len(self.health_checkers)} services")
        
        while self.running:
            await self._check_all_services()
            await asyncio.sleep(self.check_interval)

    def stop_monitoring(self):
        """Stop health monitoring"""
        self.running = False

    async def _check_all_services(self):
        """Check health of all registered services"""
        tasks = [checker.check_health() for checker in self.health_checkers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Health check failed with exception: {result}")
                continue
                
            self._store_result(result)
            self._check_for_alerts(result)

    def _store_result(self, result: HealthCheckResult):
        """Store health check result with history management"""
        history = self.health_history[result.service_name]
        history.append(result)
        
        # Keep only last 100 results
        if len(history) > 100:
            history.pop(0)

    def _check_for_alerts(self, result: HealthCheckResult):
        """Check if alerts should be triggered"""
        if result.status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]:
            for callback in self.alert_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    self.logger.error(f"Alert callback failed: {e}")

    def get_service_health(self, service_name: str) -> Optional[HealthCheckResult]:
        """Get latest health status for a service"""
        history = self.health_history.get(service_name, [])
        return history[-1] if history else None

    def get_overall_health(self) -> Dict:
        """Get overall system health summary"""
        healthy_count = 0
        degraded_count = 0
        unhealthy_count = 0
        total_services = len(self.health_checkers)
        
        for checker in self.health_checkers:
            latest_result = self.get_service_health(checker.service_name)
            if latest_result:
                if latest_result.status == HealthStatus.HEALTHY:
                    healthy_count += 1
                elif latest_result.status == HealthStatus.DEGRADED:
                    degraded_count += 1
                else:
                    unhealthy_count += 1
        
        overall_status = HealthStatus.HEALTHY
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
            
        return {
            "overall_status": overall_status.value,
            "total_services": total_services,
            "healthy_services": healthy_count,
            "degraded_services": degraded_count,
            "unhealthy_services": unhealthy_count,
            "timestamp": datetime.now().isoformat()
        }

# Alert system
def slack_alert(webhook_url: str):
    """Create Slack alert callback"""
    def alert_callback(result: HealthCheckResult):
        import requests
        
        color = "danger" if result.status == HealthStatus.UNHEALTHY else "warning"
        message = {
            "attachments": [{
                "color": color,
                "title": f"Service Health Alert: {result.service_name}",
                "fields": [
                    {"title": "Status", "value": result.status.value, "short": True},
                    {"title": "Response Time", "value": f"{result.response_time_ms:.1f}ms", "short": True},
                    {"title": "Timestamp", "value": result.timestamp.isoformat(), "short": False}
                ]
            }]
        }
        
        if result.error_message:
            message["attachments"][0]["fields"].append({
                "title": "Error", "value": result.error_message, "short": False
            })
        
        requests.post(webhook_url, json=message)
    
    return alert_callback
```

### 4. Service Discovery and Registration

```python
# service_discovery.py
import asyncio
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aiohttp
import consul

@dataclass
class ServiceInstance:
    service_name: str
    instance_id: str
    host: str
    port: int
    health_check_url: str
    metadata: Dict = None
    last_heartbeat: datetime = None

class ConsulServiceDiscovery:
    def __init__(self, consul_host: str = "localhost", consul_port: int = 8500):
        self.consul = consul.Consul(host=consul_host, port=consul_port)
        self.logger = logging.getLogger(__name__)

    def register_service(self, instance: ServiceInstance) -> bool:
        """Register a service instance with Consul"""
        try:
            self.consul.agent.service.register(
                name=instance.service_name,
                service_id=instance.instance_id,
                address=instance.host,
                port=instance.port,
                check=consul.Check.http(
                    instance.health_check_url,
                    interval="10s",
                    timeout="5s"
                ),
                meta=instance.metadata or {}
            )
            self.logger.info(f"Registered service: {instance.service_name}#{instance.instance_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register service: {e}")
            return False

    def deregister_service(self, instance_id: str) -> bool:
        """Deregister a service instance"""
        try:
            self.consul.agent.service.deregister(instance_id)
            self.logger.info(f"Deregistered service: {instance_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to deregister service: {e}")
            return False

    def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """Discover healthy instances of a service"""
        try:
            _, services = self.consul.health.service(service_name, passing=True)
            instances = []
            
            for service in services:
                service_info = service['Service']
                instances.append(ServiceInstance(
                    service_name=service_info['Service'],
                    instance_id=service_info['ID'],
                    host=service_info['Address'],
                    port=service_info['Port'],
                    health_check_url=f"http://{service_info['Address']}:{service_info['Port']}/health",
                    metadata=service_info.get('Meta', {})
                ))
            
            return instances
        except Exception as e:
            self.logger.error(f"Failed to discover services: {e}")
            return []

class LoadBalancer:
    def __init__(self, service_discovery: ConsulServiceDiscovery):
        self.service_discovery = service_discovery
        self.round_robin_counters: Dict[str, int] = {}
        
    def get_service_instance(self, service_name: str) -> Optional[ServiceInstance]:
        """Get next available service instance using round-robin"""
        instances = self.service_discovery.discover_services(service_name)
        
        if not instances:
            return None
            
        # Round-robin selection
        counter = self.round_robin_counters.get(service_name, 0)
        selected_instance = instances[counter % len(instances)]
        self.round_robin_counters[service_name] = counter + 1
        
        return selected_instance
```

### 5. Microservice Examples

Now let's implement the actual microservices that will use these resilience patterns:

#### User Service (Authentication & User Management)

```python
# user_service.py
from flask import Flask, request, jsonify
import jwt
import hashlib
import sqlite3
import logging
from datetime import datetime, timedelta
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Health check endpoint
@app.route('/health')
def health():
    try:
        # Check database connectivity
        conn = sqlite3.connect('users.db')
        conn.execute('SELECT 1')
        conn.close()
        
        return jsonify({
            'status': 'healthy',
            'service': 'user-service',
            'timestamp': datetime.now().isoformat(),
            'checks': {
                'database': 'ok'
            }
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'user-service',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 503

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')
        
        if not all([username, password, email]):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Hash password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Save to database
        conn = sqlite3.connect('users.db')
        try:
            conn.execute(
                'INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)',
                (username, password_hash, email)
            )
            conn.commit()
            user_id = conn.lastrowid
        except sqlite3.IntegrityError:
            return jsonify({'error': 'Username or email already exists'}), 409
        finally:
            conn.close()
            
        return jsonify({
            'message': 'User created successfully',
            'user_id': user_id
        }), 201
        
    except Exception as e:
        logging.error(f"Registration error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not all([username, password]):
            return jsonify({'error': 'Missing username or password'}), 400
            
        # Hash password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Check credentials
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, username, email FROM users WHERE username = ? AND password_hash = ?',
            (username, password_hash)
        )
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return jsonify({'error': 'Invalid credentials'}), 401
            
        # Generate JWT token
        token = jwt.encode({
            'user_id': user[0],
            'username': user[1],
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'token': token,
            'user': {
                'id': user[0],
                'username': user[1],
                'email': user[2]
            }
        }), 200
        
    except Exception as e:
        logging.error(f"Login error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/verify', methods=['POST'])
def verify_token():
    """Verify JWT token - used by other services"""
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if not token:
            return jsonify({'error': 'No token provided'}), 401
            
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return jsonify({
            'valid': True,
            'user_id': payload['user_id'],
            'username': payload['username']
        }), 200
        
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Token expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        logging.error(f"Token verification error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5001, debug=False)
```

#### Order Service (Business Logic)

```python
# order_service.py
import asyncio
import aiohttp
from aiohttp import web
import json
import sqlite3
import logging
from datetime import datetime
import uuid
from circuit_breaker import circuit_breaker
from retry_mechanism import retry

# Database setup
def init_db():
    conn = sqlite3.connect('orders.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            items TEXT NOT NULL,
            total_amount DECIMAL(10,2) NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

class OrderService:
    def __init__(self):
        self.payment_service_url = "http://payment-service:5003"
        self.user_service_url = "http://user-service:5001"
        self.notification_service_url = "http://notification-service:5004"

    @circuit_breaker(failure_threshold=3, recovery_timeout=30)
    @retry(max_attempts=3, base_delay=1.0)
    async def verify_user(self, token: str) -> dict:
        """Verify user token with User Service"""
        async with aiohttp.ClientSession() as session:
            headers = {'Authorization': f'Bearer {token}'}
            async with session.post(f"{self.user_service_url}/verify", headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"User verification failed: {response.status}")

    @circuit_breaker(failure_threshold=5, recovery_timeout=60)
    @retry(max_attempts=2, base_delay=2.0)
    async def process_payment(self, order_id: str, amount: float) -> dict:
        """Process payment with Payment Service"""
        async with aiohttp.ClientSession() as session:
            payload = {'order_id': order_id, 'amount': amount}
            async with session.post(f"{self.payment_service_url}/process", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Payment processing failed: {response.status}")

    async def send_notification(self, user_id: int, message: str, notification_type: str = "order"):
        """Send notification (fire-and-forget with circuit breaker)"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    'user_id': user_id,
                    'message': message,
                    'type': notification_type
                }
                async with session.post(f"{self.notification_service_url}/send", json=payload) as response:
                    if response.status != 200:
                        logging.warning(f"Notification failed: {response.status}")
        except Exception as e:
            logging.error(f"Notification error: {e}")
            # Don't fail the order if notification fails

order_service = OrderService()

async def health_check(request):
    try:
        # Check database connectivity
        conn = sqlite3.connect('orders.db')
        conn.execute('SELECT 1')
        conn.close()
        
        # Check external service connectivity
        checks = {'database': 'ok'}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{order_service.user_service_url}/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    checks['user_service'] = 'ok' if resp.status == 200 else 'degraded'
        except:
            checks['user_service'] = 'unavailable'
            
        return web.json_response({
            'status': 'healthy',
            'service': 'order-service',
            'timestamp': datetime.now().isoformat(),
            'checks': checks
        })
    except Exception as e:
        return web.json_response({
            'status': 'unhealthy',
            'service': 'order-service',
            'error': str(e)
        }, status=503)

async def create_order(request):
    try:
        # Get authorization token
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return web.json_response({'error': 'Missing or invalid authorization'}, status=401)
        
        token = auth_header.replace('Bearer ', '')
        
        # Verify user
        try:
            user_info = await order_service.verify_user(token)
            user_id = user_info['user_id']
        except Exception as e:
            return web.json_response({'error': 'User verification failed'}, status=401)
        
        # Parse order data
        data = await request.json()
        items = data.get('items', [])
        
        if not items:
            return web.json_response({'error': 'No items specified'}, status=400)
        
        # Calculate total
        total_amount = sum(item.get('price', 0) * item.get('quantity', 1) for item in items)
        
        # Create order
        order_id = str(uuid.uuid4())
        conn = sqlite3.connect('orders.db')
        conn.execute(
            'INSERT INTO orders (id, user_id, items, total_amount, status) VALUES (?, ?, ?, ?, ?)',
            (order_id, user_id, json.dumps(items), total_amount, 'pending')
        )
        conn.commit()
        conn.close()
        
        # Process payment
        try:
            payment_result = await order_service.process_payment(order_id, total_amount)
            
            # Update order status
            conn = sqlite3.connect('orders.db')
            conn.execute(
                'UPDATE orders SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
                ('confirmed', order_id)
            )
            conn.commit()
            conn.close()
            
            # Send confirmation notification (async)
            asyncio.create_task(order_service.send_notification(
                user_id,
                f"Order {order_id} confirmed! Total: ${total_amount:.2f}",
                "order_confirmation"
            ))
            
            return web.json_response({
                'order_id': order_id,
                'status': 'confirmed',
                'total_amount': total_amount,
                'payment_id': payment_result.get('payment_id')
            })
            
        except Exception as e:
            # Payment failed - update order status
            conn = sqlite3.connect('orders.db')
            conn.execute(
                'UPDATE orders SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
                ('failed', order_id)
            )
            conn.commit()
            conn.close()
            
            logging.error(f"Payment failed for order {order_id}: {e}")
            return web.json_response({
                'error': 'Payment processing failed',
                'order_id': order_id,
                'status': 'failed'
            }, status=402)
            
    except Exception as e:
        logging.error(f"Order creation error: {e}")
        return web.json_response({'error': 'Internal server error'}, status=500)

async def get_order(request):
    try:
        order_id = request.match_info['order_id']
        
        conn = sqlite3.connect('orders.db')
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, user_id, items, total_amount, status, created_at, updated_at FROM orders WHERE id = ?',
            (order_id,)
        )
        order = cursor.fetchone()
        conn.close()
        
        if not order:
            return web.json_response({'error': 'Order not found'}, status=404)
            
        return web.json_response({
            'order_id': order[0],
            'user_id': order[1],
            'items': json.loads(order[2]),
            'total_amount': float(order[3]),
            'status': order[4],
            'created_at': order[5],
            'updated_at': order[6]
        })
        
    except Exception as e:
        logging.error(f"Get order error: {e}")
        return web.json_response({'error': 'Internal server error'}, status=500)

# Setup routes
app = web.Application()
app.router.add_get('/health', health_check)
app.router.add_post('/orders', create_order)
app.router.add_get('/orders/{order_id}', get_order)

if __name__ == '__main__':
    init_db()
    web.run_app(app, host='0.0.0.0', port=5002)
```

### 6. Chaos Engineering Implementation

Chaos engineering helps identify weaknesses in your system by deliberately introducing failures.

```python
# chaos_monkey.py
import asyncio
import random
import logging
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import docker
import requests
import psutil

class ChaosType(Enum):
    NETWORK_LATENCY = "network_latency"
    SERVICE_KILL = "service_kill"
    CPU_STRESS = "cpu_stress"
    MEMORY_STRESS = "memory_stress"
    DISK_FILL = "disk_fill"
    NETWORK_PARTITION = "network_partition"

@dataclass
class ChaosExperiment:
    name: str
    chaos_type: ChaosType
    target_service: str
    duration_seconds: int
    severity: float  # 0.0 to 1.0
    conditions: Dict = None
    metadata: Dict = None

class ChaosMonkey:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.active_experiments: Dict[str, ChaosExperiment] = {}
        self.experiment_history: List[Dict] = []
        self.logger = logging.getLogger(__name__)

    async def run_experiment(self, experiment: ChaosExperiment) -> Dict:
        """Execute a chaos engineering experiment"""
        experiment_id = f"{experiment.name}_{int(time.time())}"
        
        self.logger.info(f"Starting chaos experiment: {experiment.name}")
        self.active_experiments[experiment_id] = experiment
        
        start_time = datetime.now()
        result = {
            'experiment_id': experiment_id,
            'name': experiment.name,
            'type': experiment.chaos_type.value,
            'target': experiment.target_service,
            'start_time': start_time.isoformat(),
            'status': 'running'
        }
        
        try:
            if experiment.chaos_type == ChaosType.SERVICE_KILL:
                await self._kill_service(experiment)
            elif experiment.chaos_type == ChaosType.NETWORK_LATENCY:
                await self._inject_network_latency(experiment)
            elif experiment.chaos_type == ChaosType.CPU_STRESS:
                await self._stress_cpu(experiment)
            elif experiment.chaos_type == ChaosType.MEMORY_STRESS:
                await self._stress_memory(experiment)
            elif experiment.chaos_type == ChaosType.NETWORK_PARTITION:
                await self._create_network_partition(experiment)
            
            result['status'] = 'completed'
            
        except Exception as e:
            self.logger.error(f"Chaos experiment {experiment.name} failed: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)
        
        finally:
            result['end_time'] = datetime.now().isoformat()
            result['duration'] = (datetime.now() - start_time).total_seconds()
            
            self.experiment_history.append(result)
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]
        
        return result

    async def _kill_service(self, experiment: ChaosExperiment):
        """Kill and restart a service container"""
        try:
            # Find container
            containers = self.docker_client.containers.list(
                filters={'name': experiment.target_service}
            )
            
            if not containers:
                raise Exception(f"Container {experiment.target_service} not found")
            
            container = containers[0]
            container_id = container.id
            
            self.logger.info(f"Killing container: {experiment.target_service}")
            container.kill()
            
            # Wait for specified duration
            await asyncio.sleep(experiment.duration_seconds)
            
            # Restart container
            self.logger.info(f"Restarting container: {experiment.target_service}")
            container.restart()
            
            # Wait for container to be ready
            await asyncio.sleep(10)
            
        except Exception as e:
            self.logger.error(f"Service kill experiment failed: {e}")
            raise

    async def _inject_network_latency(self, experiment: ChaosExperiment):
        """Inject network latency using tc (traffic control)"""
        # This requires the container to have tc installed
        latency_ms = int(experiment.severity * 1000)  # Convert to milliseconds
        
        try:
            container = self.docker_client.containers.get(experiment.target_service)
            
            # Add network delay
            add_latency_cmd = f"tc qdisc add dev eth0 root netem delay {latency_ms}ms"
            container.exec_run(add_latency_cmd, privileged=True)
            
            self.logger.info(f"Added {latency_ms}ms latency to {experiment.target_service}")
            
            # Wait for experiment duration
            await asyncio.sleep(experiment.duration_seconds)
            
            # Remove network delay
            remove_latency_cmd = "tc qdisc del dev eth0 root"
            container.exec_run(remove_latency_cmd, privileged=True)
            
            self.logger.info(f"Removed latency from {experiment.target_service}")
            
        except Exception as e:
            self.logger.error(f"Network latency experiment failed: {e}")
            raise

    async def _stress_cpu(self, experiment: ChaosExperiment):
        """Create CPU stress on target service"""
        cpu_cores = int(experiment.severity * psutil.cpu_count())
        
        try:
            container = self.docker_client.containers.get(experiment.target_service)
            
            # Install and run stress tool
            install_cmd = "apt-get update && apt-get install -y stress"
            container.exec_run(install_cmd)
            
            stress_cmd = f"timeout {experiment.duration_seconds} stress --cpu {cpu_cores}"
            container.exec_run(stress_cmd, detach=True)
            
            self.logger.info(f"Started CPU stress on {experiment.target_service} ({cpu_cores} cores)")
            
            # Wait for experiment to complete
            await asyncio.sleep(experiment.duration_seconds + 5)
            
        except Exception as e:
            self.logger.error(f"CPU stress experiment failed: {e}")
            raise

    async def _stress_memory(self, experiment: ChaosExperiment):
        """Create memory pressure on target service"""
        # Allocate percentage of available memory
        memory_mb = int(experiment.severity * 512)  # Up to 512MB
        
        try:
            container = self.docker_client.containers.get(experiment.target_service)
            
            stress_cmd = f"timeout {experiment.duration_seconds} stress --vm 1 --vm-bytes {memory_mb}M"
            container.exec_run(stress_cmd, detach=True)
            
            self.logger.info(f"Started memory stress on {experiment.target_service} ({memory_mb}MB)")
            
            await asyncio.sleep(experiment.duration_seconds + 5)
            
        except Exception as e:
            self.logger.error(f"Memory stress experiment failed: {e}")
            raise

    async def _create_network_partition(self, experiment: ChaosExperiment):
        """Create network partition between services"""
        try:
            # Block traffic between services using iptables
            container = self.docker_client.containers.get(experiment.target_service)
            
            # Get other service IPs from conditions
            blocked_services = experiment.conditions.get('blocked_services', [])
            
            for service in blocked_services:
                # Block outgoing traffic to service
                block_cmd = f"iptables -A OUTPUT -d {service} -j DROP"
                container.exec_run(block_cmd, privileged=True)
                
                self.logger.info(f"Blocked traffic from {experiment.target_service} to {service}")
            
            await asyncio.sleep(experiment.duration_seconds)
            
            # Remove blocks
            for service in blocked_services:
                unblock_cmd = f"iptables -D OUTPUT -d {service} -j DROP"
                container.exec_run(unblock_cmd, privileged=True)
                
                self.logger.info(f"Restored traffic from {experiment.target_service} to {service}")
            
        except Exception as e:
            self.logger.error(f"Network partition experiment failed: {e}")
            raise

class ChaosScheduler:
    def __init__(self, chaos_monkey: ChaosMonkey):
        self.chaos_monkey = chaos_monkey
        self.scheduled_experiments: List[Dict] = []
        self.running = False

    def schedule_experiment(self, experiment: ChaosExperiment, 
                          start_time: datetime, 
                          repeat_interval: Optional[timedelta] = None):
        """Schedule a chaos experiment"""
        self.scheduled_experiments.append({
            'experiment': experiment,
            'start_time': start_time,
            'repeat_interval': repeat_interval,
            'last_run': None
        })

    async def start_scheduler(self):
        """Start the chaos experiment scheduler"""
        self.running = True
        
        while self.running:
            now = datetime.now()
            
            for scheduled in self.scheduled_experiments:
                should_run = False
                
                if scheduled['last_run'] is None:
                    # First run
                    should_run = now >= scheduled['start_time']
                elif scheduled['repeat_interval']:
                    # Recurring experiment
                    next_run = scheduled['last_run'] + scheduled['repeat_interval']
                    should_run = now >= next_run
                
                if should_run:
                    self.logger.info(f"Running scheduled experiment: {scheduled['experiment'].name}")
                    try:
                        await self.chaos_monkey.run_experiment(scheduled['experiment'])
                        scheduled['last_run'] = now
                    except Exception as e:
                        self.logger.error(f"Scheduled experiment failed: {e}")
            
            await asyncio.sleep(60)  # Check every minute

    def stop_scheduler(self):
        """Stop the chaos experiment scheduler"""
        self.running = False

# Example usage and test scenarios
def create_sample_experiments() -> List[ChaosExperiment]:
    """Create sample chaos experiments for testing"""
    
    experiments = [
        # Kill payment service to test order handling
        ChaosExperiment(
            name="payment_service_outage",
            chaos_type=ChaosType.SERVICE_KILL,
            target_service="payment-service",
            duration_seconds=30,
            severity=1.0
        ),
        
        # Add latency to user service
        ChaosExperiment(
            name="user_service_latency",
            chaos_type=ChaosType.NETWORK_LATENCY,
            target_service="user-service",
            duration_seconds=60,
            severity=0.5  # 500ms latency
        ),
        
        # Stress CPU on order service
        ChaosExperiment(
            name="order_service_cpu_stress",
            chaos_type=ChaosType.CPU_STRESS,
            target_service="order-service",
            duration_seconds=120,
            severity=0.8  # 80% of available CPU
        ),
        
        # Create network partition
        ChaosExperiment(
            name="payment_isolation",
            chaos_type=ChaosType.NETWORK_PARTITION,
            target_service="order-service",
            duration_seconds=45,
            severity=1.0,
            conditions={'blocked_services': ['payment-service']}
        )
    ]
    
    return experiments

# Chaos engineering test runner
async def run_chaos_tests():
    """Run a series of chaos engineering tests"""
    chaos_monkey = ChaosMonkey()
    experiments = create_sample_experiments()
    
    results = []
    
    for experiment in experiments:
        print(f"\n🐵 Running Chaos Experiment: {experiment.name}")
        print(f"   Type: {experiment.chaos_type.value}")
        print(f"   Target: {experiment.target_service}")
        print(f"   Duration: {experiment.duration_seconds}s")
        
        result = await chaos_monkey.run_experiment(experiment)
        results.append(result)
        
        print(f"   Result: {result['status']}")
        
        # Wait between experiments
        await asyncio.sleep(30)
    
    return results

if __name__ == "__main__":
    asyncio.run(run_chaos_tests())
```

### 7. Complete Docker Setup

Let's create a complete Docker environment for our resilient microservices platform:

#### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Service Discovery
  consul:
    image: consul:1.15
    command: consul agent -dev -ui -client=0.0.0.0 -bootstrap-expect=1
    ports:
      - "8500:8500"
    environment:
      - CONSUL_BIND_INTERFACE=eth0
    networks:
      - microservices

  # User Service
  user-service:
    build:
      context: ./user-service
      dockerfile: Dockerfile
    ports:
      - "5001:5001"
    environment:
      - SECRET_KEY=super-secret-key-change-in-production
      - CONSUL_HOST=consul
    depends_on:
      - consul
    networks:
      - microservices
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Order Service
  order-service:
    build:
      context: ./order-service
      dockerfile: Dockerfile
    ports:
      - "5002:5002"
    environment:
      - USER_SERVICE_URL=http://user-service:5001
      - PAYMENT_SERVICE_URL=http://payment-service:5003
      - NOTIFICATION_SERVICE_URL=http://notification-service:5004
    depends_on:
      - user-service
      - consul
    networks:
      - microservices
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5002/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Payment Service
  payment-service:
    build:
      context: ./payment-service
      dockerfile: Dockerfile
    ports:
      - "5003:5003"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/payments
    depends_on:
      - postgres
      - consul
    networks:
      - microservices
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5003/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Notification Service
  notification-service:
    build:
      context: ./notification-service
      dockerfile: Dockerfile
    ports:
      - "5004:5004"
    environment:
      - REDIS_URL=redis://redis:6379
      - SMTP_HOST=mailhog
      - SMTP_PORT=1025
    depends_on:
      - redis
      - mailhog
      - consul
    networks:
      - microservices

  # PostgreSQL Database
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=payments
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - microservices

  # Redis for caching and queues
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - microservices

  # Mail server for testing notifications
  mailhog:
    image: mailhog/mailhog
    ports:
      - "1025:1025"
      - "8025:8025"  # Web UI
    networks:
      - microservices

  # API Gateway
  api-gateway:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - user-service
      - order-service
      - payment-service
      - notification-service
    networks:
      - microservices

  # Monitoring
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    networks:
      - microservices

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus
    networks:
      - microservices

  # Load Balancer
  haproxy:
    image: haproxy:2.8
    ports:
      - "8080:8080"
      - "8404:8404"  # Stats page
    volumes:
      - ./haproxy/haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg
    depends_on:
      - order-service
      - payment-service
    networks:
      - microservices

volumes:
  postgres_data:
  prometheus_data:
  grafana_data:

networks:
  microservices:
    driver: bridge
```

#### Service Dockerfiles

```dockerfile
# user-service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5001/health || exit 1

EXPOSE 5001

CMD ["python", "user_service.py"]
```

```dockerfile
# order-service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including traffic control tools
RUN apt-get update && apt-get install -y \
    curl \
    iproute2 \
    iptables \
    stress \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5002/health || exit 1

EXPOSE 5002

CMD ["python", "order_service.py"]
```

#### HAProxy Configuration

```
# haproxy/haproxy.cfg
global
    daemon
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog

# Stats page
stats enable
stats uri /stats
stats refresh 30s
stats admin if TRUE

# Frontend - Load Balancer Entry Point
frontend api_frontend
    bind *:8080
    
    # Health check endpoint
    acl is_health_check path_beg /health
    use_backend health_backend if is_health_check
    
    # Route to appropriate backends
    acl is_orders path_beg /orders
    acl is_payments path_beg /payments
    
    use_backend order_backend if is_orders
    use_backend payment_backend if is_payments
    
    default_backend order_backend

# Backend - Order Service
backend order_backend
    balance roundrobin
    option httpchk GET /health
    
    server order1 order-service:5002 check
    # Add more instances for scaling:
    # server order2 order-service-2:5002 check
    # server order3 order-service-3:5002 check

# Backend - Payment Service  
backend payment_backend
    balance roundrobin
    option httpchk GET /health
    
    server payment1 payment-service:5003 check

# Backend - Health checks
backend health_backend
    server health1 127.0.0.1:8404
```

#### Nginx API Gateway Configuration

```nginx
# nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream user_service {
        least_conn;
        server user-service:5001 max_fails=3 fail_timeout=30s;
        # Add more instances for scaling
    }
    
    upstream order_service {
        least_conn;
        server order-service:5002 max_fails=3 fail_timeout=30s;
    }
    
    upstream payment_service {
        least_conn;
        server payment-service:5003 max_fails=3 fail_timeout=30s;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for" '
                    'rt=$request_time uct="$upstream_connect_time" '
                    'uht="$upstream_header_time" urt="$upstream_response_time"';

    server {
        listen 80;
        server_name api.microservices.local;
        
        access_log /var/log/nginx/access.log main;
        error_log /var/log/nginx/error.log;

        # Global rate limiting
        limit_req zone=api burst=20 nodelay;

        # Health check endpoint
        location /health {
            access_log off;
            return 200 "OK\n";
            add_header Content-Type text/plain;
        }

        # User service routes
        location /api/v1/users/ {
            proxy_pass http://user_service/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 5s;
            proxy_send_timeout 10s;
            proxy_read_timeout 10s;
            
            # Circuit breaker simulation
            proxy_next_upstream error timeout invalid_header http_500 http_502 http_503 http_504;
        }

        # Order service routes
        location /api/v1/orders/ {
            proxy_pass http://order_service/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 5s;
            proxy_send_timeout 30s;  # Longer timeout for order processing
            proxy_read_timeout 30s;
        }

        # Payment service routes
        location /api/v1/payments/ {
            proxy_pass http://payment_service/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 5s;
            proxy_send_timeout 15s;
            proxy_read_timeout 15s;
        }

        # Default route
        location / {
            return 404 "Not Found\n";
        }
    }
}
```

## Testing and Validation

### 1. Automated Testing Strategy

#### Unit Tests for Resilience Components

```python
# tests/test_circuit_breaker.py
import pytest
import time
from unittest.mock import Mock
from circuit_breaker import CircuitBreaker, CircuitState

class TestCircuitBreaker:
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in normal operation"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        
        # Mock successful function
        mock_func = Mock(return_value="success")
        
        result = cb.call(mock_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        
        # Mock failing function
        mock_func = Mock(side_effect=Exception("Service unavailable"))
        
        # First 2 failures should keep circuit closed
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call(mock_func)
            assert cb.state == CircuitState.CLOSED
        
        # Third failure should open circuit
        with pytest.raises(Exception):
            cb.call(mock_func)
        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery mechanism"""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        # Mock function that fails then succeeds
        mock_func = Mock(side_effect=[Exception("fail"), Exception("fail"), "success"])
        
        # Trigger failures to open circuit
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call(mock_func)
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Next call should enter half-open and succeed
        result = cb.call(mock_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

# tests/test_retry_mechanism.py
import pytest
import time
from unittest.mock import Mock, patch
from retry_mechanism import RetryMechanism, RetryConfig

class TestRetryMechanism:
    def test_successful_first_attempt(self):
        """Test no retry needed for successful first attempt"""
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        retry_mechanism = RetryMechanism(config)
        
        mock_func = Mock(return_value="success")
        
        result = retry_mechanism.execute(mock_func)
        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_on_failure(self):
        """Test retry mechanism with eventual success"""
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        retry_mechanism = RetryMechanism(config)
        
        # Fail twice, then succeed
        mock_func = Mock(side_effect=[Exception("fail"), Exception("fail"), "success"])
        
        result = retry_mechanism.execute(mock_func)
        assert result == "success"
        assert mock_func.call_count == 3

    def test_exponential_backoff(self):
        """Test exponential backoff delay calculation"""
        config = RetryConfig(max_attempts=3, base_delay=1.0, exponential_base=2.0, jitter=False)
        retry_mechanism = RetryMechanism(config)
        
        # Test delay calculation
        assert retry_mechanism._calculate_delay(0) == 1.0  # 1 * 2^0
        assert retry_mechanism._calculate_delay(1) == 2.0  # 1 * 2^1
        assert retry_mechanism._calculate_delay(2) == 4.0  # 1 * 2^2

    def test_max_attempts_exceeded(self):
        """Test failure after exceeding max attempts"""
        config = RetryConfig(max_attempts=2, base_delay=0.1)
        retry_mechanism = RetryMechanism(config)
        
        mock_func = Mock(side_effect=Exception("always fail"))
        
        with pytest.raises(Exception, match="always fail"):
            retry_mechanism.execute(mock_func)
        
        assert mock_func.call_count == 2
```

#### Integration Tests

```python
# tests/test_integration.py
import pytest
import asyncio
import aiohttp
import docker
from testcontainers.compose import DockerCompose

class TestMicroservicesIntegration:
    @pytest.fixture(scope="class")
    def docker_compose(self):
        """Setup Docker Compose environment for testing"""
        compose = DockerCompose(".", compose_file_name="docker-compose.test.yml")
        compose.start()
        
        # Wait for services to be ready
        asyncio.run(self._wait_for_services())
        
        yield compose
        compose.stop()

    async def _wait_for_services(self, max_wait=60):
        """Wait for all services to be healthy"""
        services = [
            "http://localhost:5001/health",  # user-service
            "http://localhost:5002/health",  # order-service
            "http://localhost:5003/health",  # payment-service
        ]
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            all_healthy = True
            
            async with aiohttp.ClientSession() as session:
                for service_url in services:
                    try:
                        async with session.get(service_url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                            if response.status != 200:
                                all_healthy = False
                                break
                    except:
                        all_healthy = False
                        break
            
            if all_healthy:
                return
                
            await asyncio.sleep(2)
        
        raise Exception("Services failed to become healthy within timeout")

    @pytest.mark.asyncio
    async def test_user_registration_and_login(self, docker_compose):
        """Test user registration and login flow"""
        async with aiohttp.ClientSession() as session:
            # Register user
            user_data = {
                "username": "testuser",
                "password": "testpass123",
                "email": "test@example.com"
            }
            
            async with session.post("http://localhost:5001/register", json=user_data) as response:
                assert response.status == 201
                result = await response.json()
                assert "user_id" in result

            # Login user
            login_data = {"username": "testuser", "password": "testpass123"}
            async with session.post("http://localhost:5001/login", json=login_data) as response:
                assert response.status == 200
                result = await response.json()
                assert "token" in result
                return result["token"]

    @pytest.mark.asyncio
    async def test_order_creation_flow(self, docker_compose):
        """Test complete order creation with payment"""
        # First get auth token
        token = await self.test_user_registration_and_login(docker_compose)
        
        async with aiohttp.ClientSession() as session:
            # Create order
            order_data = {
                "items": [
                    {"name": "Test Product", "price": 29.99, "quantity": 2}
                ]
            }
            
            headers = {"Authorization": f"Bearer {token}"}
            async with session.post("http://localhost:5002/orders", json=order_data, headers=headers) as response:
                assert response.status == 200
                result = await response.json()
                assert "order_id" in result
                assert result["status"] == "confirmed"
                assert result["total_amount"] == 59.98

    @pytest.mark.asyncio
    async def test_circuit_breaker_behavior(self, docker_compose):
        """Test circuit breaker opens when payment service fails"""
        # Get auth token
        token = await self.test_user_registration_and_login(docker_compose)
        
        # Stop payment service to trigger failures
        docker_client = docker.from_env()
        payment_container = docker_client.containers.get("payment-service")
        payment_container.stop()
        
        try:
            async with aiohttp.ClientSession() as session:
                order_data = {"items": [{"name": "Test", "price": 10.0, "quantity": 1}]}
                headers = {"Authorization": f"Bearer {token}"}
                
                # Make multiple requests to trigger circuit breaker
                for _ in range(5):
                    async with session.post("http://localhost:5002/orders", json=order_data, headers=headers) as response:
                        # Should fail due to payment service being down
                        assert response.status in [402, 500, 503]
                        
                        # Add small delay between requests
                        await asyncio.sleep(1)
        finally:
            # Restart payment service
            payment_container.start()
            await asyncio.sleep(10)  # Wait for service to be ready
```

### 2. Load Testing

```python
# load_test.py
import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class LoadTestResult:
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    error_rate: float

class LoadTester:
    def __init__(self, base_url: str, auth_token: str = None):
        self.base_url = base_url
        self.auth_token = auth_token
        self.response_times: List[float] = []
        self.successful_requests = 0
        self.failed_requests = 0

    async def make_request(self, session: aiohttp.ClientSession, endpoint: str, method: str = "GET", data: Dict = None):
        """Make a single HTTP request and record metrics"""
        start_time = time.time()
        
        try:
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            url = f"{self.base_url}{endpoint}"
            
            if method == "GET":
                async with session.get(url, headers=headers) as response:
                    await response.read()
                    success = response.status < 400
            elif method == "POST":
                async with session.post(url, json=data, headers=headers) as response:
                    await response.read()
                    success = response.status < 400
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
                
        except Exception as e:
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            self.failed_requests += 1

    async def run_load_test(self, endpoint: str, concurrent_users: int, requests_per_user: int, method: str = "GET", data: Dict = None) -> LoadTestResult:
        """Run load test with specified parameters"""
        
        print(f"Starting load test:")
        print(f"  Endpoint: {endpoint}")
        print(f"  Concurrent users: {concurrent_users}")
        print(f"  Requests per user: {requests_per_user}")
        print(f"  Total requests: {concurrent_users * requests_per_user}")
        
        start_time = time.time()
        
        # Create semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def user_session():
            async with semaphore:
                connector = aiohttp.TCPConnector(limit=100)
                async with aiohttp.ClientSession(connector=connector) as session:
                    tasks = []
                    for _ in range(requests_per_user):
                        task = self.make_request(session, endpoint, method, data)
                        tasks.append(task)
                    
                    await asyncio.gather(*tasks)
        
        # Create tasks for all users
        user_tasks = [user_session() for _ in range(concurrent_users)]
        
        # Run all user sessions concurrently
        await asyncio.gather(*user_tasks)
        
        total_time = time.time() - start_time
        total_requests = self.successful_requests + self.failed_requests
        
        # Calculate metrics
        result = LoadTestResult(
            total_requests=total_requests,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            average_response_time=statistics.mean(self.response_times) if self.response_times else 0,
            min_response_time=min(self.response_times) if self.response_times else 0,
            max_response_time=max(self.response_times) if self.response_times else 0,
            requests_per_second=total_requests / total_time if total_time > 0 else 0,
            error_rate=(self.failed_requests / total_requests * 100) if total_requests > 0 else 0
        )
        
        return result

async def run_comprehensive_load_tests():
    """Run comprehensive load tests on all endpoints"""
    
    # First, get authentication token
    async with aiohttp.ClientSession() as session:
        # Register and login user
        user_data = {
            "username": f"loadtest_{int(time.time())}",
            "password": "testpass123",
            "email": f"loadtest_{int(time.time())}@example.com"
        }
        
        async with session.post("http://localhost:5001/register", json=user_data) as response:
            await response.json()
        
        login_data = {"username": user_data["username"], "password": user_data["password"]}
        async with session.post("http://localhost:5001/login", json=login_data) as response:
            result = await response.json()
            auth_token = result["token"]
    
    # Test scenarios
    scenarios = [
        {
            "name": "Health Check Load Test",
            "endpoint": "/health",
            "method": "GET",
            "concurrent_users": 50,
            "requests_per_user": 20,
            "auth_required": False
        },
        {
            "name": "User Login Load Test",
            "endpoint": "/login",
            "method": "POST",
            "data": {"username": user_data["username"], "password": user_data["password"]},
            "concurrent_users": 20,
            "requests_per_user": 10,
            "auth_required": False
        },
        {
            "name": "Order Creation Load Test",
            "endpoint": "/orders",
            "method": "POST",
            "data": {"items": [{"name": "Load Test Product", "price": 19.99, "quantity": 1}]},
            "concurrent_users": 10,
            "requests_per_user": 5,
            "auth_required": True
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"Running: {scenario['name']}")
        print(f"{'='*50}")
        
        # Setup load tester
        base_url = "http://localhost:5002" if "orders" in scenario["endpoint"] else "http://localhost:5001"
        token = auth_token if scenario["auth_required"] else None
        
        tester = LoadTester(base_url, token)
        
        # Run test
        result = await tester.run_load_test(
            endpoint=scenario["endpoint"],
            concurrent_users=scenario["concurrent_users"],
            requests_per_user=scenario["requests_per_user"],
            method=scenario["method"],
            data=scenario.get("data")
        )
        
        results[scenario["name"]] = result
        
        # Print results
        print(f"\nResults for {scenario['name']}:")
        print(f"  Total Requests: {result.total_requests}")
        print(f"  Successful: {result.successful_requests}")
        print(f"  Failed: {result.failed_requests}")
        print(f"  Error Rate: {result.error_rate:.2f}%")
        print(f"  Avg Response Time: {result.average_response_time:.3f}s")
        print(f"  Min Response Time: {result.min_response_time:.3f}s")
        print(f"  Max Response Time: {result.max_response_time:.3f}s")
        print(f"  Requests/sec: {result.requests_per_second:.2f}")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_comprehensive_load_tests())
```

### 3. Monitoring and Observability

#### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # User Service
  - job_name: 'user-service'
    static_configs:
      - targets: ['user-service:5001']
    metrics_path: '/metrics'
    scrape_interval: 10s

  # Order Service
  - job_name: 'order-service'
    static_configs:
      - targets: ['order-service:5002']
    metrics_path: '/metrics'
    scrape_interval: 10s

  # Payment Service
  - job_name: 'payment-service'
    static_configs:
      - targets: ['payment-service:5003']
    metrics_path: '/metrics'
    scrape_interval: 10s

  # Infrastructure monitoring
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  # HAProxy stats
  - job_name: 'haproxy'
    static_configs:
      - targets: ['haproxy:8404']
```

#### Alert Rules

```yaml
# monitoring/alert_rules.yml
groups:
- name: microservices_alerts
  rules:
  
  # Service availability alerts
  - alert: ServiceDown
    expr: up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Service {{ $labels.instance }} is down"
      description: "Service {{ $labels.instance }} has been down for more than 1 minute."

  # High error rate alerts
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) * 100 > 5
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate on {{ $labels.instance }}"
      description: "Error rate is {{ $value }}% on {{ $labels.instance }}"

  # Circuit breaker alerts
  - alert: CircuitBreakerOpen
    expr: circuit_breaker_state > 0
    for: 30s
    labels:
      severity: warning
    annotations:
      summary: "Circuit breaker open for {{ $labels.name }}"
      description: "Circuit breaker {{ $labels.name }} has been open for more than 30 seconds"

  # High response time alerts
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
    for: 3m
    labels:
      severity: warning
    annotations:
      summary: "High response time on {{ $labels.instance }}"
      description: "95th percentile response time is {{ $value }}s on {{ $labels.instance }}"

  # Database connection alerts
  - alert: DatabaseConnectionFailure
    expr: increase(database_connection_errors_total[5m]) > 5
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Database connection failures on {{ $labels.instance }}"
      description: "{{ $value }} database connection failures in the last 5 minutes"
```

#### Grafana Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "Resilient Microservices Dashboard",
    "tags": ["microservices", "resilience"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Service Health Overview",
        "type": "stat",
        "targets": [
          {
            "expr": "up",
            "legendFormat": "{{ instance }}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "green", "value": 1}
              ]
            }
          }
        }
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{ instance }}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m]) * 100",
            "legendFormat": "{{ instance }}"
          }
        ],
        "yAxes": [
          {
            "unit": "percent"
          }
        ]
      },
      {
        "title": "Circuit Breaker Status",
        "type": "stat",
        "targets": [
          {
            "expr": "circuit_breaker_state",
            "legendFormat": "{{ name }}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "mappings": [
              {"value": 0, "text": "CLOSED", "color": "green"},
              {"value": 1, "text": "OPEN", "color": "red"},
              {"value": 2, "text": "HALF_OPEN", "color": "yellow"}
            ]
          }
        }
      },
      {
        "title": "Response Time (95th percentile)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "{{ instance }}"
          }
        ],
        "yAxes": [
          {
            "unit": "s"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "10s"
  }
}
```

## Project Deliverables

### 1. Implementation Checklist

**Core Resilience Patterns:**
- [ ] Circuit Breaker implementation with configurable thresholds
- [ ] Retry mechanism with exponential backoff and jitter
- [ ] Health check system with multiple check types
- [ ] Service discovery and registration
- [ ] Load balancing with multiple strategies

**Microservices:**
- [ ] User Service (authentication/authorization)
- [ ] Order Service (business logic)
- [ ] Payment Service (external integration simulation)
- [ ] Notification Service (async processing)

**Infrastructure:**
- [ ] Docker containerization for all services
- [ ] Docker Compose orchestration
- [ ] API Gateway (Nginx) configuration
- [ ] Load Balancer (HAProxy) setup
- [ ] Service mesh considerations

**Monitoring & Observability:**
- [ ] Prometheus metrics collection
- [ ] Grafana dashboards
- [ ] Alert rules configuration
- [ ] Distributed tracing setup
- [ ] Log aggregation

**Chaos Engineering:**
- [ ] Chaos Monkey implementation
- [ ] Automated fault injection
- [ ] Experiment scheduler
- [ ] Results analysis tools

**Testing:**
- [ ] Unit tests for resilience components
- [ ] Integration tests
- [ ] Load testing suite
- [ ] Chaos engineering tests
- [ ] Performance benchmarks

### 2. Performance Benchmarks

Your implementation should achieve:

- **Availability**: 99.9% uptime during normal operations
- **Resilience**: System remains operational with 1 service failure
- **Response Time**: 95th percentile < 500ms under normal load
- **Throughput**: Handle 1000 requests/second per service
- **Recovery Time**: < 30 seconds to detect and recover from failures
- **Error Rate**: < 1% under normal conditions

### 3. Documentation Requirements

**Architecture Documentation:**
- System architecture diagrams
- Service interaction flows
- Failure scenarios and recovery procedures
- Deployment and scaling strategies

**Technical Documentation:**
- API documentation for all services
- Configuration management guide
- Monitoring and alerting runbooks
- Chaos engineering experiment catalog

**Operational Documentation:**
- Deployment procedures
- Troubleshooting guides
- Performance tuning recommendations
- Disaster recovery procedures

## Learning Outcomes

Upon completing this project, you will have gained expertise in:

1. **Resilience Patterns**: Deep understanding of circuit breakers, retries, bulkheads, and timeouts
2. **System Design**: Experience designing fault-tolerant distributed systems
3. **Monitoring**: Implementation of comprehensive observability solutions
4. **Testing**: Chaos engineering and resilience testing methodologies
5. **DevOps**: Container orchestration and service mesh technologies
6. **Production Operations**: Real-world operational concerns and solutions

## Next Steps

After completing this project, consider exploring:

1. **Service Mesh**: Implement Istio or Linkerd for advanced traffic management
2. **Event-Driven Architecture**: Add message queues and event sourcing
3. **Multi-Region Deployment**: Implement cross-region resilience
4. **Advanced Monitoring**: Add distributed tracing with Jaeger or Zipkin
5. **Security**: Implement mTLS, OAuth2, and security scanning
6. **GitOps**: Implement CI/CD pipelines with automated deployment

This project provides a solid foundation for building production-grade microservices that can handle real-world operational challenges and failures gracefully.
